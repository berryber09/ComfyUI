"""Background asset seeder with thread management and cancellation support."""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

from app.assets.scanner import (
    RootType,
    build_asset_specs,
    collect_paths_for_roots,
    get_all_known_prefixes,
    get_prefixes_for_root,
    insert_asset_specs,
    prune_orphans_safely,
    sync_root_safely,
)
from app.database.db import dependencies_available

if TYPE_CHECKING:
    pass


class State(Enum):
    """Seeder state machine states."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"


@dataclass
class Progress:
    """Progress information for a scan operation."""

    scanned: int = 0
    total: int = 0
    created: int = 0
    skipped: int = 0


@dataclass
class ScanStatus:
    """Current status of the asset seeder."""

    state: State
    progress: Progress | None
    errors: list[str] = field(default_factory=list)


ProgressCallback = Callable[[Progress], None]


class AssetSeeder:
    """Singleton class managing background asset scanning.

    Thread-safe singleton that spawns ephemeral daemon threads for scanning.
    Each scan creates a new thread that exits when complete.
    """

    _instance: "AssetSeeder | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "AssetSeeder":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._progress: Progress | None = None
        self._errors: list[str] = []
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._roots: tuple[RootType, ...] = ()
        self._progress_callback: ProgressCallback | None = None

    def start(
        self,
        roots: tuple[RootType, ...] = ("models", "input", "output"),
        progress_callback: ProgressCallback | None = None,
        prune_first: bool = False,
    ) -> bool:
        """Start a background scan for the given roots.

        Args:
            roots: Tuple of root types to scan (models, input, output)
            progress_callback: Optional callback called with progress updates
            prune_first: If True, prune orphaned assets before scanning

        Returns:
            True if scan was started, False if already running
        """
        with self._lock:
            if self._state != State.IDLE:
                return False
            self._state = State.RUNNING
            self._progress = Progress()
            self._errors = []
            self._roots = roots
            self._prune_first = prune_first
            self._progress_callback = progress_callback
            self._cancel_event.clear()
            self._thread = threading.Thread(
                target=self._run_scan,
                name="AssetSeeder",
                daemon=True,
            )
            self._thread.start()
            return True

    def cancel(self) -> bool:
        """Request cancellation of the current scan.

        Returns:
            True if cancellation was requested, False if not running
        """
        with self._lock:
            if self._state != State.RUNNING:
                return False
            self._state = State.CANCELLING
            self._cancel_event.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the current scan to complete.

        Args:
            timeout: Maximum seconds to wait, or None for no timeout

        Returns:
            True if scan completed, False if timeout expired or no scan running
        """
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def get_status(self) -> ScanStatus:
        """Get the current status and progress of the seeder."""
        with self._lock:
            return ScanStatus(
                state=self._state,
                progress=Progress(
                    scanned=self._progress.scanned,
                    total=self._progress.total,
                    created=self._progress.created,
                    skipped=self._progress.skipped,
                )
                if self._progress
                else None,
                errors=list(self._errors),
            )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown: cancel any running scan and wait for thread.

        Args:
            timeout: Maximum seconds to wait for thread to exit
        """
        self.cancel()
        self.wait(timeout=timeout)
        with self._lock:
            self._thread = None

    def prune_orphans(self) -> int:
        """Prune orphaned assets that are outside all known root prefixes.

        This operation is decoupled from scanning to prevent partial scans
        from accidentally deleting assets belonging to other roots.

        Should be called explicitly when cleanup is desired, typically after
        a full scan of all roots or during maintenance.

        Returns:
            Number of orphaned assets pruned, or 0 if dependencies unavailable
            or a scan is currently running
        """
        with self._lock:
            if self._state != State.IDLE:
                logging.warning("Cannot prune orphans while scan is running")
                return 0

        if not dependencies_available():
            logging.warning(
                "Database dependencies not available, skipping orphan pruning"
            )
            return 0

        all_prefixes = get_all_known_prefixes()
        pruned = prune_orphans_safely(all_prefixes)
        if pruned > 0:
            logging.info("Pruned %d orphaned assets", pruned)
        return pruned

    def _is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event.is_set()

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit a WebSocket event if server is available."""
        try:
            from server import PromptServer

            if hasattr(PromptServer, "instance") and PromptServer.instance:
                PromptServer.instance.send_sync(event_type, data)
        except Exception:
            pass

    def _update_progress(
        self,
        scanned: int | None = None,
        total: int | None = None,
        created: int | None = None,
        skipped: int | None = None,
    ) -> None:
        """Update progress counters (thread-safe)."""
        with self._lock:
            if self._progress is None:
                return
            if scanned is not None:
                self._progress.scanned = scanned
            if total is not None:
                self._progress.total = total
            if created is not None:
                self._progress.created = created
            if skipped is not None:
                self._progress.skipped = skipped
            if self._progress_callback:
                try:
                    self._progress_callback(
                        Progress(
                            scanned=self._progress.scanned,
                            total=self._progress.total,
                            created=self._progress.created,
                            skipped=self._progress.skipped,
                        )
                    )
                except Exception:
                    pass

    def _add_error(self, message: str) -> None:
        """Add an error message (thread-safe)."""
        with self._lock:
            self._errors.append(message)

    def _log_scan_config(self, roots: tuple[RootType, ...]) -> None:
        """Log the directories that will be scanned."""
        import folder_paths

        for root in roots:
            if root == "models":
                logging.info(
                    "Asset scan [models] directory: %s",
                    os.path.abspath(folder_paths.models_dir),
                )
            else:
                prefixes = get_prefixes_for_root(root)
                if prefixes:
                    logging.info("Asset scan [%s] directories: %s", root, prefixes)

    def _run_scan(self) -> None:
        """Main scan loop running in background thread."""
        t_start = time.perf_counter()
        roots = self._roots
        cancelled = False
        total_created = 0
        skipped_existing = 0
        total_paths = 0

        try:
            if not dependencies_available():
                self._add_error("Database dependencies not available")
                self._emit_event(
                    "assets.seed.error",
                    {"message": "Database dependencies not available"},
                )
                return

            if self._prune_first:
                all_prefixes = get_all_known_prefixes()
                pruned = prune_orphans_safely(all_prefixes)
                if pruned > 0:
                    logging.info("Pruned %d orphaned assets before scan", pruned)

            if self._is_cancelled():
                logging.info("Asset scan cancelled after pruning phase")
                cancelled = True
                return

            self._log_scan_config(roots)

            existing_paths: set[str] = set()
            for r in roots:
                if self._is_cancelled():
                    logging.info("Asset scan cancelled during sync phase")
                    cancelled = True
                    return
                existing_paths.update(sync_root_safely(r))

            if self._is_cancelled():
                logging.info("Asset scan cancelled after sync phase")
                cancelled = True
                return

            paths = collect_paths_for_roots(roots)
            total_paths = len(paths)
            self._update_progress(total=total_paths)

            self._emit_event(
                "assets.seed.started",
                {"roots": list(roots), "total": total_paths},
            )

            specs, tag_pool, skipped_existing = build_asset_specs(paths, existing_paths)
            self._update_progress(skipped=skipped_existing)

            if self._is_cancelled():
                logging.info("Asset scan cancelled after building specs")
                cancelled = True
                return

            batch_size = 500
            last_progress_time = time.perf_counter()
            progress_interval = 1.0

            for i in range(0, len(specs), batch_size):
                if self._is_cancelled():
                    logging.info(
                        "Asset scan cancelled after %d/%d files (created=%d)",
                        i,
                        len(specs),
                        total_created,
                    )
                    cancelled = True
                    return

                batch = specs[i : i + batch_size]
                batch_tags = {t for spec in batch for t in spec["tags"]}
                try:
                    created = insert_asset_specs(batch, batch_tags)
                    total_created += created
                except Exception as e:
                    self._add_error(f"Batch insert failed at offset {i}: {e}")
                    logging.exception("Batch insert failed at offset %d", i)

                scanned = i + len(batch)
                self._update_progress(scanned=scanned, created=total_created)

                now = time.perf_counter()
                if now - last_progress_time >= progress_interval:
                    self._emit_event(
                        "assets.seed.progress",
                        {
                            "scanned": scanned,
                            "total": len(specs),
                            "created": total_created,
                        },
                    )
                    last_progress_time = now

            self._update_progress(scanned=len(specs), created=total_created)

            elapsed = time.perf_counter() - t_start
            logging.info(
                "Asset scan(roots=%s) completed in %.3fs (created=%d, skipped=%d, total=%d)",
                roots,
                elapsed,
                total_created,
                skipped_existing,
                len(paths),
            )

            self._emit_event(
                "assets.seed.completed",
                {
                    "scanned": len(specs),
                    "total": total_paths,
                    "created": total_created,
                    "skipped": skipped_existing,
                    "elapsed": round(elapsed, 3),
                },
            )

        except Exception as e:
            self._add_error(f"Scan failed: {e}")
            logging.exception("Asset scan failed")
            self._emit_event("assets.seed.error", {"message": str(e)})
        finally:
            if cancelled:
                self._emit_event(
                    "assets.seed.cancelled",
                    {
                        "scanned": self._progress.scanned if self._progress else 0,
                        "total": total_paths,
                        "created": total_created,
                    },
                )
            with self._lock:
                self._state = State.IDLE


asset_seeder = AssetSeeder()
