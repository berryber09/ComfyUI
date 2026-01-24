import pytest
import torch
import tempfile
import os
import av
from fractions import Fraction
from comfy_api.input_impl.video_types import (
    VideoFromFile,
    VideoFromComponents,
    SliceOp,
)
from comfy_api.util.video_types import VideoComponents


def create_test_video(width=4, height=4, frames=10, fps=30):
    """Helper to create a temporary video file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    with av.open(tmp.name, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for i in range(frames):
            frame_data = torch.ones(height, width, 3, dtype=torch.uint8) * (i * 25)
            frame = av.VideoFrame.from_ndarray(frame_data.numpy(), format="rgb24")
            frame = frame.reformat(format="yuv420p")
            packet = stream.encode(frame)
            container.mux(packet)

        packet = stream.encode(None)
        container.mux(packet)

    return tmp.name


@pytest.fixture
def video_file_10_frames():
    file_path = create_test_video(frames=10)
    yield file_path
    os.unlink(file_path)


@pytest.fixture
def video_components_10_frames():
    images = torch.rand(10, 4, 4, 3)
    return VideoComponents(images=images, frame_rate=Fraction(30))


class TestSliceOp:
    def test_apply_slices_correctly(self, video_components_10_frames):
        op = SliceOp(start_frame=2, frame_count=3)
        result = op.apply(video_components_10_frames)

        assert result.images.shape[0] == 3
        assert torch.equal(result.images, video_components_10_frames.images[2:5])

    def test_compute_frame_count(self):
        op = SliceOp(start_frame=2, frame_count=5)
        assert op.compute_frame_count(10) == 5

    def test_compute_frame_count_clamps(self):
        op = SliceOp(start_frame=8, frame_count=5)
        assert op.compute_frame_count(10) == 2


class TestVideoSliced:
    def test_sliced_returns_new_instance(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(2, 3)

        assert video is not sliced
        assert len(video._operations) == 0
        assert len(sliced._operations) == 1

    def test_get_components_applies_operations(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(2, 3)

        components = sliced.get_components()

        assert components.images.shape[0] == 3
        assert torch.equal(components.images, video_components_10_frames.images[2:5])

    def test_get_frame_count(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(2, 3)

        assert sliced.get_frame_count() == 3

    def test_get_duration(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(0, 3)

        assert sliced.get_duration() == pytest.approx(0.1)

    def test_chained_slices_compose(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(2, 6).sliced(1, 3)

        components = sliced.get_components()

        assert components.images.shape[0] == 3
        assert torch.equal(components.images, video_components_10_frames.images[3:6])

    def test_operations_list_is_immutable(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced1 = video.sliced(0, 5)
        sliced2 = sliced1.sliced(1, 2)

        assert len(video._operations) == 0
        assert len(sliced1._operations) == 1
        assert len(sliced2._operations) == 2

    def test_from_file(self, video_file_10_frames):
        video = VideoFromFile(video_file_10_frames)
        sliced = video.sliced(2, 3)

        components = sliced.get_components()

        assert components.images.shape[0] == 3
        assert sliced.get_frame_count() == 3

    def test_save_sliced_video(self, video_components_10_frames, tmp_path):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(2, 3)

        output_path = str(tmp_path / "sliced_output.mp4")
        sliced.save_to(output_path)

        saved_video = VideoFromFile(output_path)
        assert saved_video.get_frame_count() == 3

    def test_materialization_clears_ops(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(2, 3)

        assert len(sliced._operations) == 1
        sliced.get_components()
        assert len(sliced._operations) == 0

    def test_second_get_components_uses_cache(self, video_components_10_frames):
        video = VideoFromComponents(video_components_10_frames)
        sliced = video.sliced(2, 3)

        first = sliced.get_components()
        second = sliced.get_components()

        assert first.images.shape == second.images.shape
        assert torch.equal(first.images, second.images)
