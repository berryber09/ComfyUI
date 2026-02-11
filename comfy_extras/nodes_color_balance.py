from typing_extensions import override
import torch

from comfy_api.latest import ComfyExtension, io, ui


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class ColorBalanceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorBalance",
            display_name="Color Balance",
            category="image/adjustment",
            inputs=[
                io.Image.Input("image"),
                io.ColorBalance.Input("settings"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, settings: dict) -> io.NodeOutput:
        shadows_red = settings.get("shadows_red", 0)
        shadows_green = settings.get("shadows_green", 0)
        shadows_blue = settings.get("shadows_blue", 0)
        midtones_red = settings.get("midtones_red", 0)
        midtones_green = settings.get("midtones_green", 0)
        midtones_blue = settings.get("midtones_blue", 0)
        highlights_red = settings.get("highlights_red", 0)
        highlights_green = settings.get("highlights_green", 0)
        highlights_blue = settings.get("highlights_blue", 0)

        result = image.clone().float()

        # Compute per-pixel luminance
        luminance = (
            0.2126 * result[..., 0]
            + 0.7152 * result[..., 1]
            + 0.0722 * result[..., 2]
        )

        # Compute tonal range weights
        shadow_weight = 1.0 - _smoothstep(0.0, 0.5, luminance)
        highlight_weight = _smoothstep(0.5, 1.0, luminance)
        midtone_weight = 1.0 - shadow_weight - highlight_weight

        # Apply offsets per channel
        for ch, (s, m, h) in enumerate([
            (shadows_red, midtones_red, highlights_red),
            (shadows_green, midtones_green, highlights_green),
            (shadows_blue, midtones_blue, highlights_blue),
        ]):
            offset = (
                shadow_weight * (s / 100.0)
                + midtone_weight * (m / 100.0)
                + highlight_weight * (h / 100.0)
            )
            result[..., ch] = result[..., ch] + offset

        result = torch.clamp(result, 0, 1)
        return io.NodeOutput(result, ui=ui.PreviewImage(result))


class ColorBalanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColorBalanceNode]


async def comfy_entrypoint() -> ColorBalanceExtension:
    return ColorBalanceExtension()
