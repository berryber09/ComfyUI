from typing_extensions import override
import torch
import numpy as np

from comfy_api.latest import ComfyExtension, io, ui


def _monotone_cubic_hermite(xs, ys, x_query):
    """Evaluate monotone cubic Hermite interpolation at x_query points."""
    n = len(xs)
    if n == 0:
        return np.zeros_like(x_query)
    if n == 1:
        return np.full_like(x_query, ys[0])

    # Compute slopes
    deltas = np.diff(ys) / np.maximum(np.diff(xs), 1e-10)

    # Compute tangents (Fritsch-Carlson)
    slopes = np.zeros(n)
    slopes[0] = deltas[0]
    slopes[-1] = deltas[-1]
    for i in range(1, n - 1):
        if deltas[i - 1] * deltas[i] <= 0:
            slopes[i] = 0
        else:
            slopes[i] = (deltas[i - 1] + deltas[i]) / 2

    # Enforce monotonicity
    for i in range(n - 1):
        if deltas[i] == 0:
            slopes[i] = 0
            slopes[i + 1] = 0
        else:
            alpha = slopes[i] / deltas[i]
            beta = slopes[i + 1] / deltas[i]
            s = alpha ** 2 + beta ** 2
            if s > 9:
                t = 3 / np.sqrt(s)
                slopes[i] = t * alpha * deltas[i]
                slopes[i + 1] = t * beta * deltas[i]

    # Evaluate
    result = np.zeros_like(x_query, dtype=np.float64)
    indices = np.searchsorted(xs, x_query, side='right') - 1
    indices = np.clip(indices, 0, n - 2)

    for i in range(n - 1):
        mask = indices == i
        if not np.any(mask):
            continue
        dx = xs[i + 1] - xs[i]
        if dx == 0:
            result[mask] = ys[i]
            continue
        t = (x_query[mask] - xs[i]) / dx
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        result[mask] = h00 * ys[i] + h10 * dx * slopes[i] + h01 * ys[i + 1] + h11 * dx * slopes[i + 1]

    # Clamp edges
    result[x_query <= xs[0]] = ys[0]
    result[x_query >= xs[-1]] = ys[-1]

    return result


def _build_lut(points):
    """Build a 256-entry LUT from curve control points in [0,1] space."""
    if not points or len(points) < 2:
        return np.arange(256, dtype=np.float64) / 255.0

    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)

    x_query = np.linspace(0, 1, 256)
    lut = _monotone_cubic_hermite(xs, ys, x_query)
    return np.clip(lut, 0, 1)


class ColorCurvesNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorCurves",
            display_name="Color Curves",
            category="image/adjustment",
            inputs=[
                io.Image.Input("image"),
                io.ColorCurves.Input("settings"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, settings: dict) -> io.NodeOutput:
        rgb_pts = settings.get("rgb", [[0, 0], [1, 1]])
        red_pts = settings.get("red", [[0, 0], [1, 1]])
        green_pts = settings.get("green", [[0, 0], [1, 1]])
        blue_pts = settings.get("blue", [[0, 0], [1, 1]])

        rgb_lut = _build_lut(rgb_pts)
        red_lut = _build_lut(red_pts)
        green_lut = _build_lut(green_pts)
        blue_lut = _build_lut(blue_pts)

        # Convert to numpy for LUT application
        img_np = image.cpu().numpy().copy()

        # Apply per-channel curves then RGB master curve
        for ch, ch_lut in enumerate([red_lut, green_lut, blue_lut]):
            # Per-channel curve
            indices = np.clip(img_np[..., ch] * 255, 0, 255).astype(np.int32)
            img_np[..., ch] = ch_lut[indices]
            # RGB master curve
            indices = np.clip(img_np[..., ch] * 255, 0, 255).astype(np.int32)
            img_np[..., ch] = rgb_lut[indices]

        result = torch.from_numpy(np.clip(img_np, 0, 1)).to(image.device, dtype=image.dtype)
        return io.NodeOutput(result, ui=ui.PreviewImage(result))


class ColorCurvesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColorCurvesNode]


async def comfy_entrypoint() -> ColorCurvesExtension:
    return ColorCurvesExtension()
