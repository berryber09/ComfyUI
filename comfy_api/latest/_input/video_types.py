from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union, IO
import copy
import io
import av
from .._util import VideoContainer, VideoCodec, VideoComponents


class VideoOp(ABC):
    """Base class for lazy video operations."""

    @abstractmethod
    def apply(self, components: VideoComponents) -> VideoComponents:
        pass

    @abstractmethod
    def compute_frame_count(self, input_frame_count: int) -> int:
        pass


@dataclass(frozen=True)
class SliceOp(VideoOp):
    """Extract a range of frames from the video."""
    start_frame: int
    frame_count: int

    def apply(self, components: VideoComponents) -> VideoComponents:
        total = components.images.shape[0]
        start = max(0, min(self.start_frame, total))
        end = min(start + self.frame_count, total)
        return VideoComponents(
            images=components.images[start:end],
            audio=components.audio,
            frame_rate=components.frame_rate,
            metadata=getattr(components, 'metadata', None),
        )

    def compute_frame_count(self, input_frame_count: int) -> int:
        start = max(0, min(self.start_frame, input_frame_count))
        return min(self.frame_count, input_frame_count - start)


class VideoInput(ABC):
    """
    Abstract base class for video input types.
    """

    @abstractmethod
    def get_components(self) -> VideoComponents:
        """
        Abstract method to get the video components (images, audio, and frame rate).

        Returns:
            VideoComponents containing images, audio, and frame rate
        """
        pass

    def sliced(self, start_frame: int, frame_count: int) -> "VideoInput":
        """Return a copy of this video with a slice operation appended."""
        new = copy.copy(self)
        new._operations = getattr(self, '_operations', []) + [SliceOp(start_frame, frame_count)]
        return new

    @abstractmethod
    def save_to(
        self,
        path: Union[str, IO[bytes]],
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None
    ):
        """
        Abstract method to save the video input to a file.
        """
        pass

    def get_stream_source(self) -> Union[str, io.BytesIO]:
        """
        Get a streamable source for the video. This allows processing without
        loading the entire video into memory.

        Returns:
            Either a file path (str) or a BytesIO object that can be opened with av.

        Default implementation creates a BytesIO buffer, but subclasses should
        override this for better performance when possible.
        """
        buffer = io.BytesIO()
        self.save_to(buffer)
        buffer.seek(0)
        return buffer

    # Provide a default implementation, but subclasses can provide optimized versions
    # if possible.
    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        components = self.get_components()
        return components.images.shape[2], components.images.shape[1]

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        components = self.get_components()
        frame_count = components.images.shape[0]
        return float(frame_count / components.frame_rate)

    def get_frame_count(self) -> int:
        """
        Returns the number of frames in the video.

        Default implementation uses :meth:`get_components`, which may require
        loading all frames into memory. File-based implementations should
        override this method and use container/stream metadata instead.

        Returns:
            Total number of frames as an integer.
        """
        return int(self.get_components().images.shape[0])

    def get_frame_rate(self) -> Fraction:
        """
        Returns the frame rate of the video.

        Default implementation materializes the video into memory via
        `get_components()`. Subclasses that can inspect the underlying
        container (e.g. `VideoFromFile`) should override this with a more
        efficient implementation.

        Returns:
            Frame rate as a Fraction.
        """
        return self.get_components().frame_rate

    def get_container_format(self) -> str:
        """
        Returns the container format of the video (e.g., 'mp4', 'mov', 'avi').

        Returns:
            Container format as string
        """
        # Default implementation - subclasses should override for better performance
        source = self.get_stream_source()
        with av.open(source, mode="r") as container:
            return container.format.name
