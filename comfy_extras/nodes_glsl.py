import os

from comfy.cli_args import args

if args.cpu:
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
elif not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import re
import logging
from typing import TypedDict

import numpy as np
import torch

import nodes
from comfy_api.latest import ComfyExtension, io, ui
from typing_extensions import override
from utils.install_util import get_missing_requirements_message

logger = logging.getLogger(__name__)

try:
    import glfw
    import OpenGL.GL as gl
except ImportError as e:
    raise RuntimeError(
        f"OpenGL dependencies not available.\n{get_missing_requirements_message()}\n"
        "Install with: pip install PyOpenGL PyOpenGL-accelerate glfw"
    ) from e
except AttributeError as e:
    # This happens when PyOpenGL can't load the requested platform (e.g., OSMesa not installed)
    platform = os.environ.get("PYOPENGL_PLATFORM", "default")
    if platform == "osmesa":
        raise RuntimeError(
            "OSMesa (software rendering) requested but not installed.\n"
            "OSMesa is required for --cpu mode.\n\n"
            "Install OSMesa:\n"
            "  e.g. Ubuntu/Debian: sudo apt install libosmesa6-dev\n"
            "Or disable CPU mode to use hardware rendering."
        ) from e
    elif platform == "egl":
        raise RuntimeError(
            "EGL (headless rendering) requested but not available.\n"
            "EGL is used for headless GPU rendering without a display.\n\n"
            "Install EGL:\n"
            "  e.g. Ubuntu/Debian: sudo apt install libegl1-mesa-dev libgles2-mesa-dev\n"
            "Or set DISPLAY/WAYLAND_DISPLAY environment variable if you have a display."
        ) from e
    else:
        raise RuntimeError(
            f"OpenGL initialization failed (platform: {platform}).\n"
            "Ensure OpenGL drivers are installed and working."
        ) from e


class SizeModeInput(TypedDict):
    size_mode: str
    width: int
    height: int


MAX_IMAGES = 5      # u_image0-4
MAX_UNIFORMS = 5    # u_float0-4, u_int0-4
MAX_OUTPUTS = 4     # fragColor0-3 (MRT)

# Vertex shader using gl_VertexID trick - no VBO needed.
# Draws a single triangle that covers the entire screen:
#
#     (-1,3)
#       /|
#      / |  <- visible area is the unit square from (-1,-1) to (1,1)
#     /  |     parts outside get clipped away
# (-1,-1)---(3,-1)
#
# v_texCoord is computed from clip space: * 0.5 + 0.5 maps (-1,1) -> (0,1)
VERTEX_SHADER = """#version 330 core
out vec2 v_texCoord;
void main() {
    vec2 verts[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    v_texCoord = verts[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
}
"""

DEFAULT_FRAGMENT_SHADER = """#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform vec2 u_resolution;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

void main() {
    fragColor0 = texture(u_image0, v_texCoord);
}
"""


def _convert_es_to_desktop(source: str) -> str:
    """Convert GLSL ES (WebGL) shader source to desktop GLSL 330 core."""
    # Remove any existing #version directive
    source = re.sub(r"#version\s+\d+(\s+es)?\s*\n?", "", source, flags=re.IGNORECASE)
    # Remove precision qualifiers (not needed in desktop GLSL)
    source = re.sub(r"precision\s+(lowp|mediump|highp)\s+\w+\s*;\s*\n?", "", source)
    # Prepend desktop GLSL version
    return "#version 330 core\n" + source


class GLContext:
    """Manages OpenGL context and resources for shader execution."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if GLContext._initialized:
            return
        GLContext._initialized = True

        import time
        start = time.perf_counter()

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self._window = glfw.create_window(64, 64, "ComfyUI GLSL", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)

        # Create VAO (required for core profile even if we don't use vertex attributes)
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        elapsed = (time.perf_counter() - start) * 1000

        # Log device info
        renderer = gl.glGetString(gl.GL_RENDERER)
        vendor = gl.glGetString(gl.GL_VENDOR)
        version = gl.glGetString(gl.GL_VERSION)
        renderer = renderer.decode() if renderer else "Unknown"
        vendor = vendor.decode() if vendor else "Unknown"
        version = version.decode() if version else "Unknown"

        logger.info(f"GLSL context initialized in {elapsed:.1f}ms - {renderer} ({vendor}), GL {version}")

    def make_current(self):
        glfw.make_context_current(self._window)
        gl.glBindVertexArray(self._vao)


def _compile_shader(source: str, shader_type: int) -> int:
    """Compile a shader and return its ID."""
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)

    if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        error = gl.glGetShaderInfoLog(shader).decode()
        gl.glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation failed:\n{error}")

    return shader


def _create_program(vertex_source: str, fragment_source: str) -> int:
    """Create and link a shader program."""
    vertex_shader = _compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
    try:
        fragment_shader = _compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
    except RuntimeError:
        gl.glDeleteShader(vertex_shader)
        raise

    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        error = gl.glGetProgramInfoLog(program).decode()
        gl.glDeleteProgram(program)
        raise RuntimeError(f"Program linking failed:\n{error}")

    return program


def _render_shader_batch(
    fragment_code: str,
    width: int,
    height: int,
    image_batches: list[list[np.ndarray]],
    floats: list[float],
    ints: list[int],
) -> list[list[np.ndarray]]:
    """
    Render a fragment shader for multiple batches efficiently.

    Compiles shader once, reuses framebuffer/textures across batches.

    Args:
        fragment_code: User's fragment shader code
        width: Output width
        height: Output height
        image_batches: List of batches, each batch is a list of input images (H, W, C) float32 [0,1]
        floats: List of float uniforms
        ints: List of int uniforms

    Returns:
        List of batch outputs, each is a list of output images (H, W, 4) float32 [0,1]
    """
    if not image_batches:
        return []

    ctx = GLContext()
    ctx.make_current()

    # Convert from GLSL ES to desktop GLSL 330
    fragment_source = _convert_es_to_desktop(fragment_code)

    # Track resources for cleanup
    program = None
    fbo = None
    output_textures = []
    input_textures = []

    num_inputs = len(image_batches[0])

    try:
        # Compile shaders (once for all batches)
        try:
            program = _create_program(VERTEX_SHADER, fragment_source)
        except RuntimeError:
            logger.error(f"Fragment shader:\n{fragment_source}")
            raise

        gl.glUseProgram(program)

        # Create framebuffer with multiple color attachments (reused for all batches)
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        draw_buffers = []
        for i in range(MAX_OUTPUTS):
            tex = gl.glGenTextures(1)
            output_textures.append(tex)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0 + i, gl.GL_TEXTURE_2D, tex, 0)
            draw_buffers.append(gl.GL_COLOR_ATTACHMENT0 + i)

        gl.glDrawBuffers(MAX_OUTPUTS, draw_buffers)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete")

        # Create input textures (reused for all batches)
        for i in range(num_inputs):
            tex = gl.glGenTextures(1)
            input_textures.append(tex)
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            loc = gl.glGetUniformLocation(program, f"u_image{i}")
            if loc >= 0:
                gl.glUniform1i(loc, i)

        # Set static uniforms (once for all batches)
        loc = gl.glGetUniformLocation(program, "u_resolution")
        if loc >= 0:
            gl.glUniform2f(loc, float(width), float(height))

        for i, v in enumerate(floats):
            loc = gl.glGetUniformLocation(program, f"u_float{i}")
            if loc >= 0:
                gl.glUniform1f(loc, v)

        for i, v in enumerate(ints):
            loc = gl.glGetUniformLocation(program, f"u_int{i}")
            if loc >= 0:
                gl.glUniform1i(loc, v)

        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_BLEND)  # Ensure no alpha blending - write output directly

        # Process each batch
        all_batch_outputs = []
        for images in image_batches:
            # Update input textures with this batch's images
            for i, img in enumerate(images):
                gl.glActiveTexture(gl.GL_TEXTURE0 + i)
                gl.glBindTexture(gl.GL_TEXTURE_2D, input_textures[i])

                # Flip vertically for GL coordinates, ensure RGBA
                img_flipped = np.ascontiguousarray(img[::-1, :, :])
                if img_flipped.shape[2] == 3:
                    img_flipped = np.ascontiguousarray(np.concatenate(
                        [img_flipped, np.ones((*img_flipped.shape[:2], 1), dtype=np.float32)],
                        axis=2,
                    ))

                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, img_flipped.shape[1], img_flipped.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, img_flipped)

            # Render
            gl.glClearColor(0, 0, 0, 0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

            # Read back outputs for this batch
            batch_outputs = []
            for tex in output_textures:
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_FLOAT)
                img = np.frombuffer(data, dtype=np.float32).reshape(height, width, 4)
                batch_outputs.append(np.ascontiguousarray(img[::-1, :, :]))

            all_batch_outputs.append(batch_outputs)

        return all_batch_outputs

    finally:
        # Unbind before deleting
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)

        if input_textures:
            gl.glDeleteTextures(len(input_textures), input_textures)
        if output_textures:
            gl.glDeleteTextures(len(output_textures), output_textures)
        if fbo is not None:
            gl.glDeleteFramebuffers(1, [fbo])
        if program is not None:
            gl.glDeleteProgram(program)

class GLSLShader(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        image_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"),
            prefix="image",
            min=1,
            max=MAX_IMAGES,
        )

        float_template = io.Autogrow.TemplatePrefix(
            io.Float.Input("float", default=0.0),
            prefix="u_float",
            min=0,
            max=MAX_UNIFORMS,
        )

        int_template = io.Autogrow.TemplatePrefix(
            io.Int.Input("int", default=0),
            prefix="u_int",
            min=0,
            max=MAX_UNIFORMS,
        )

        return io.Schema(
            node_id="GLSLShader",
            display_name="GLSL Shader",
            category="image/shader",
            description=(
                f"Apply GLSL fragment shaders to images. "
                f"Inputs: u_image0-{MAX_IMAGES-1} (sampler2D), u_resolution (vec2), "
                f"u_float0-{MAX_UNIFORMS-1}, u_int0-{MAX_UNIFORMS-1}. "
                f"Outputs: layout(location = 0-{MAX_OUTPUTS-1}) out vec4 fragColor0-{MAX_OUTPUTS-1}."
            ),
            inputs=[
                io.String.Input(
                    "fragment_shader",
                    default=DEFAULT_FRAGMENT_SHADER,
                    multiline=True,
                    tooltip="GLSL fragment shader source code (GLSL ES 3.00 / WebGL 2.0 compatible)",
                ),
                io.DynamicCombo.Input(
                    "size_mode",
                    options=[
                        io.DynamicCombo.Option("from_input", []),
                        io.DynamicCombo.Option(
                            "custom",
                            [
                                io.Int.Input(
                                    "width",
                                    default=512,
                                    min=1,
                                    max=nodes.MAX_RESOLUTION,
                                ),
                                io.Int.Input(
                                    "height",
                                    default=512,
                                    min=1,
                                    max=nodes.MAX_RESOLUTION,
                                ),
                            ],
                        ),
                    ],
                    tooltip="Output size: 'from_input' uses first input image dimensions, 'custom' allows manual size",
                ),
                io.Autogrow.Input("images", template=image_template),
                io.Autogrow.Input("floats", template=float_template),
                io.Autogrow.Input("ints", template=int_template),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE0"),
                io.Image.Output(display_name="IMAGE1"),
                io.Image.Output(display_name="IMAGE2"),
                io.Image.Output(display_name="IMAGE3"),
            ],
        )

    @classmethod
    def execute(
        cls,
        fragment_shader: str,
        size_mode: SizeModeInput,
        images: io.Autogrow.Type,
        floats: io.Autogrow.Type = None,
        ints: io.Autogrow.Type = None,
        **kwargs,
    ) -> io.NodeOutput:
        image_list = [v for v in images.values() if v is not None]
        float_list = (
            [v if v is not None else 0.0 for v in floats.values()] if floats else []
        )
        int_list = [v if v is not None else 0 for v in ints.values()] if ints else []

        if not image_list:
            raise ValueError("At least one input image is required")

        # Determine output dimensions
        if size_mode["size_mode"] == "custom":
            out_width = size_mode["width"]
            out_height = size_mode["height"]
        else:
            out_height, out_width = image_list[0].shape[1:3]

        batch_size = image_list[0].shape[0]

        # Prepare batches
        image_batches = []
        for batch_idx in range(batch_size):
            batch_images = [img_tensor[batch_idx].cpu().numpy().astype(np.float32) for img_tensor in image_list]
            image_batches.append(batch_images)

        all_batch_outputs = _render_shader_batch(
            fragment_shader,
            out_width,
            out_height,
            image_batches,
            float_list,
            int_list,
        )

        # Collect outputs into tensors
        all_outputs = [[] for _ in range(MAX_OUTPUTS)]
        for batch_outputs in all_batch_outputs:
            for i, out_img in enumerate(batch_outputs):
                all_outputs[i].append(torch.from_numpy(out_img))

        output_tensors = [torch.stack(all_outputs[i], dim=0) for i in range(MAX_OUTPUTS)]
        return io.NodeOutput(
            *output_tensors,
            ui=cls._build_ui_output(image_list, output_tensors[0]),
        )

    @classmethod
    def _build_ui_output(
        cls, image_list: list[torch.Tensor], output_batch: torch.Tensor
    ) -> dict[str, list]:
        """Build UI output with input and output images for client-side shader execution."""
        combined_inputs = torch.cat(image_list, dim=0)
        input_images_ui = ui.ImageSaveHelper.save_images(
            combined_inputs,
            filename_prefix="GLSLShader_input",
            folder_type=io.FolderType.temp,
            cls=None,
            compress_level=1,
        )

        output_images_ui = ui.ImageSaveHelper.save_images(
            output_batch,
            filename_prefix="GLSLShader_output",
            folder_type=io.FolderType.temp,
            cls=None,
            compress_level=1,
        )

        return {"input_images": input_images_ui, "images": output_images_ui}


class GLSLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GLSLShader]


async def comfy_entrypoint() -> GLSLExtension:
    return GLSLExtension()
