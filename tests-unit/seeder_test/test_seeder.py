"""Unit tests for the AssetSeeder background scanning class."""

import threading
import time
from unittest.mock import patch

import pytest

from app.assets.seeder import AssetSeeder, Progress, ScanPhase, State


@pytest.fixture
def fresh_seeder():
    """Create a fresh AssetSeeder instance for testing (bypasses singleton)."""
    seeder = object.__new__(AssetSeeder)
    seeder._initialized = False
    seeder.__init__()
    yield seeder
    seeder.shutdown(timeout=1.0)


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for isolated testing."""
    with (
        patch("app.assets.seeder.dependencies_available", return_value=True),
        patch("app.assets.seeder.sync_root_safely", return_value=set()),
        patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
        patch("app.assets.seeder.build_stub_specs", return_value=([], set(), 0)),
        patch("app.assets.seeder.insert_asset_specs", return_value=0),
        patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
        patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
    ):
        yield


class TestSeederStateTransitions:
    """Test state machine transitions."""

    def test_initial_state_is_idle(self, fresh_seeder: AssetSeeder):
        assert fresh_seeder.get_status().state == State.IDLE

    def test_start_transitions_to_running(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        started = fresh_seeder.start(roots=("models",))
        assert started is True
        status = fresh_seeder.get_status()
        assert status.state in (State.RUNNING, State.IDLE)

    def test_start_while_running_returns_false(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            time.sleep(0.05)

            second_start = fresh_seeder.start(roots=("models",))
            assert second_start is False

            barrier.set()

    def test_cancel_transitions_to_cancelling(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            time.sleep(0.05)

            cancelled = fresh_seeder.cancel()
            assert cancelled is True
            assert fresh_seeder.get_status().state == State.CANCELLING

            barrier.set()

    def test_cancel_when_idle_returns_false(self, fresh_seeder: AssetSeeder):
        cancelled = fresh_seeder.cancel()
        assert cancelled is False

    def test_state_returns_to_idle_after_completion(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        fresh_seeder.start(roots=("models",))
        completed = fresh_seeder.wait(timeout=5.0)
        assert completed is True
        assert fresh_seeder.get_status().state == State.IDLE


class TestSeederWait:
    """Test wait() behavior."""

    def test_wait_blocks_until_complete(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        fresh_seeder.start(roots=("models",))
        completed = fresh_seeder.wait(timeout=5.0)
        assert completed is True
        assert fresh_seeder.get_status().state == State.IDLE

    def test_wait_returns_false_on_timeout(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=10.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            completed = fresh_seeder.wait(timeout=0.1)
            assert completed is False

            barrier.set()

    def test_wait_when_idle_returns_true(self, fresh_seeder: AssetSeeder):
        completed = fresh_seeder.wait(timeout=1.0)
        assert completed is True


class TestSeederProgress:
    """Test progress tracking."""

    def test_get_status_returns_progress_during_scan(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        progress_seen = []
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=5.0)
            return ["/path/file1.safetensors", "/path/file2.safetensors"]

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            time.sleep(0.05)

            status = fresh_seeder.get_status()
            assert status.progress is not None
            progress_seen.append(status.progress)

            barrier.set()

    def test_progress_callback_is_invoked(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        progress_updates: list[Progress] = []

        def callback(p: Progress):
            progress_updates.append(p)

        with patch(
            "app.assets.seeder.collect_paths_for_roots",
            return_value=[f"/path/file{i}.safetensors" for i in range(10)],
        ):
            fresh_seeder.start(roots=("models",), progress_callback=callback)
            fresh_seeder.wait(timeout=5.0)

        assert len(progress_updates) > 0


class TestSeederCancellation:
    """Test cancellation behavior."""

    def test_scan_commits_partial_progress_on_cancellation(
        self, fresh_seeder: AssetSeeder
    ):
        insert_count = 0
        barrier = threading.Event()

        def slow_insert(specs, tags):
            nonlocal insert_count
            insert_count += 1
            if insert_count >= 2:
                barrier.wait(timeout=5.0)
            return len(specs)

        paths = [f"/path/file{i}.safetensors" for i in range(1500)]
        specs = [
            {
                "abs_path": p,
                "size_bytes": 100,
                "mtime_ns": 0,
                "info_name": f"file{i}",
                "tags": [],
                "fname": f"file{i}",
                "metadata": None,
                "hash": None,
                "mime_type": None,
            }
            for i, p in enumerate(paths)
        ]

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=paths),
            patch(
                "app.assets.seeder.build_stub_specs", return_value=(specs, set(), 0)
            ),
            patch("app.assets.seeder.insert_asset_specs", side_effect=slow_insert),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",))
            time.sleep(0.1)

            fresh_seeder.cancel()
            barrier.set()
            fresh_seeder.wait(timeout=5.0)

            assert insert_count >= 1


class TestSeederErrorHandling:
    """Test error handling behavior."""

    def test_database_errors_captured_in_status(self, fresh_seeder: AssetSeeder):
        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch(
                "app.assets.seeder.collect_paths_for_roots",
                return_value=["/path/file.safetensors"],
            ),
            patch(
                "app.assets.seeder.build_stub_specs",
                return_value=(
                    [
                        {
                            "abs_path": "/path/file.safetensors",
                            "size_bytes": 100,
                            "mtime_ns": 0,
                            "info_name": "file",
                            "tags": [],
                            "fname": "file",
                            "metadata": None,
                            "hash": None,
                            "mime_type": None,
                        }
                    ],
                    set(),
                    0,
                ),
            ),
            patch(
                "app.assets.seeder.insert_asset_specs",
                side_effect=Exception("DB connection failed"),
            ),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            status = fresh_seeder.get_status()
            assert len(status.errors) > 0
            assert "DB connection failed" in status.errors[0]

    def test_dependencies_unavailable_captured_in_errors(
        self, fresh_seeder: AssetSeeder
    ):
        with patch("app.assets.seeder.dependencies_available", return_value=False):
            fresh_seeder.start(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            status = fresh_seeder.get_status()
            assert len(status.errors) > 0
            assert "dependencies" in status.errors[0].lower()

    def test_thread_crash_resets_state_to_idle(self, fresh_seeder: AssetSeeder):
        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch(
                "app.assets.seeder.sync_root_safely",
                side_effect=RuntimeError("Unexpected crash"),
            ),
        ):
            fresh_seeder.start(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            status = fresh_seeder.get_status()
            assert status.state == State.IDLE
            assert len(status.errors) > 0


class TestSeederThreadSafety:
    """Test thread safety of concurrent operations."""

    def test_concurrent_start_calls_spawn_only_one_thread(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            results = []

            def try_start():
                results.append(fresh_seeder.start(roots=("models",)))

            threads = [threading.Thread(target=try_start) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            barrier.set()

            assert sum(results) == 1

    def test_get_status_safe_during_scan(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))

            statuses = []
            for _ in range(100):
                statuses.append(fresh_seeder.get_status())
                time.sleep(0.001)

            barrier.set()

            assert all(
                s.state in (State.RUNNING, State.IDLE, State.CANCELLING)
                for s in statuses
            )


class TestSeederMarkMissing:
    """Test mark_missing_outside_prefixes behavior."""

    def test_mark_missing_when_idle(self, fresh_seeder: AssetSeeder):
        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch(
                "app.assets.seeder.get_all_known_prefixes",
                return_value=["/models", "/input", "/output"],
            ),
            patch(
                "app.assets.seeder.mark_missing_outside_prefixes_safely", return_value=5
            ) as mock_mark,
        ):
            result = fresh_seeder.mark_missing_outside_prefixes()
            assert result == 5
            mock_mark.assert_called_once_with(["/models", "/input", "/output"])

    def test_mark_missing_returns_zero_when_running(
        self, fresh_seeder: AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            time.sleep(0.05)

            result = fresh_seeder.mark_missing_outside_prefixes()
            assert result == 0

            barrier.set()

    def test_mark_missing_returns_zero_when_dependencies_unavailable(
        self, fresh_seeder: AssetSeeder
    ):
        with patch("app.assets.seeder.dependencies_available", return_value=False):
            result = fresh_seeder.mark_missing_outside_prefixes()
            assert result == 0

    def test_prune_first_flag_triggers_mark_missing_before_scan(
        self, fresh_seeder: AssetSeeder
    ):
        call_order = []

        def track_mark(prefixes):
            call_order.append("mark_missing")
            return 3

        def track_sync(root):
            call_order.append(f"sync_{root}")
            return set()

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.get_all_known_prefixes", return_value=["/models"]),
            patch("app.assets.seeder.mark_missing_outside_prefixes_safely", side_effect=track_mark),
            patch("app.assets.seeder.sync_root_safely", side_effect=track_sync),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_stub_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",), prune_first=True)
            fresh_seeder.wait(timeout=5.0)

            assert call_order[0] == "mark_missing"
            assert "sync_models" in call_order


class TestSeederPhases:
    """Test phased scanning behavior."""

    def test_start_fast_only_runs_fast_phase(self, fresh_seeder: AssetSeeder):
        """Verify start_fast only runs the fast phase."""
        fast_called = []
        enrich_called = []

        def track_fast(*args, **kwargs):
            fast_called.append(True)
            return ([], set(), 0)

        def track_enrich(*args, **kwargs):
            enrich_called.append(True)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_stub_specs", side_effect=track_fast),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=track_enrich),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start_fast(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            assert len(fast_called) == 1
            assert len(enrich_called) == 0

    def test_start_enrich_only_runs_enrich_phase(self, fresh_seeder: AssetSeeder):
        """Verify start_enrich only runs the enrich phase."""
        fast_called = []
        enrich_called = []

        def track_fast(*args, **kwargs):
            fast_called.append(True)
            return ([], set(), 0)

        def track_enrich(*args, **kwargs):
            enrich_called.append(True)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_stub_specs", side_effect=track_fast),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=track_enrich),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start_enrich(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            assert len(fast_called) == 0
            assert len(enrich_called) == 1

    def test_full_scan_runs_both_phases(self, fresh_seeder: AssetSeeder):
        """Verify full scan runs both fast and enrich phases."""
        fast_called = []
        enrich_called = []

        def track_fast(*args, **kwargs):
            fast_called.append(True)
            return ([], set(), 0)

        def track_enrich(*args, **kwargs):
            enrich_called.append(True)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_stub_specs", side_effect=track_fast),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=track_enrich),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",), phase=ScanPhase.FULL)
            fresh_seeder.wait(timeout=5.0)

            assert len(fast_called) == 1
            assert len(enrich_called) == 1
