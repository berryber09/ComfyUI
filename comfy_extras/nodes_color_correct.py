from typing_extensions import override
import torch
import numpy as np

from comfy_api.latest import ComfyExtension, io, ui


class ColorCorrectNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorCorrect",
            display_name="Color Correct",
            category="image/adjustment",
            inputs=[
                io.Image.Input("image"),
                io.ColorCorrect.Input("settings"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, settings: dict) -> io.NodeOutput:
        temperature = settings.get("temperature", 0)
        hue = settings.get("hue", 0)
        brightness = settings.get("brightness", 0)
        contrast = settings.get("contrast", 0)
        saturation = settings.get("saturation", 0)
        gamma = settings.get("gamma", 1.0)

        result = image.clone()

        # Brightness: scale RGB values
        if brightness != 0:
            factor = 1.0 + brightness / 100.0
            result = result * factor

        # Contrast: adjust around midpoint
        if contrast != 0:
            factor = 1.0 + contrast / 100.0
            mean = result[..., :3].mean()
            result[..., :3] = (result[..., :3] - mean) * factor + mean

        # Temperature: shift warm (red+) / cool (blue+)
        if temperature != 0:
            temp_factor = temperature / 100.0
            result[..., 0] = result[..., 0] + temp_factor * 0.1  # Red
            result[..., 2] = result[..., 2] - temp_factor * 0.1  # Blue

        # Gamma correction
        if gamma != 1.0:
            result[..., :3] = torch.pow(torch.clamp(result[..., :3], 0, 1), 1.0 / gamma)

        # Saturation: convert to HSV-like space
        if saturation != 0:
            factor = 1.0 + saturation / 100.0
            gray = result[..., :3].mean(dim=-1, keepdim=True)
            result[..., :3] = gray + (result[..., :3] - gray) * factor

        # Hue rotation: rotate in RGB space using rotation matrix
        if hue != 0:
            angle = np.radians(hue)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # Rodrigues' rotation formula around (1,1,1)/sqrt(3) axis
            k = 1.0 / 3.0
            rotation = torch.tensor([
                [cos_a + k * (1 - cos_a), k * (1 - cos_a) - sin_a / np.sqrt(3), k * (1 - cos_a) + sin_a / np.sqrt(3)],
                [k * (1 - cos_a) + sin_a / np.sqrt(3), cos_a + k * (1 - cos_a), k * (1 - cos_a) - sin_a / np.sqrt(3)],
                [k * (1 - cos_a) - sin_a / np.sqrt(3), k * (1 - cos_a) + sin_a / np.sqrt(3), cos_a + k * (1 - cos_a)]
            ], dtype=result.dtype, device=result.device)
            rgb = result[..., :3]
            result[..., :3] = torch.matmul(rgb, rotation.T)

        result = torch.clamp(result, 0, 1)
        return io.NodeOutput(result, ui=ui.PreviewImage(result))


class ColorCorrectExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColorCorrectNode]


async def comfy_entrypoint() -> ColorCorrectExtension:
    return ColorCorrectExtension()
