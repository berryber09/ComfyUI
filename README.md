# ComfyUI Fork — Bypass LoRA Chain Fix

Fork of [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) with fixes for bypass LoRA chaining on INT8 quantized models.

## What This Fork Changes

Two files modified from upstream:

### `comfy/sd.py` — Shared Bypass Manager via Attachments

**Problem:** When multiple `LoraLoaderBypass` nodes are chained (e.g. distilled LoRA + style LoRA), the original code creates unique injection keys (`bypass_lora_1`, `bypass_lora_2`). This causes separate inject/eject cycles on the same modules. On second generation (execution cache hit), the eject order corrupts the forward chain — hook1 restores `real_forward`, then hook2 restores `hook1._bypass_forward`, leaving `module.forward = hook1._bypass_forward`. Next inject, hook1 saves its own `_bypass_forward` as `original_forward` and calls itself recursively.

**Fix:** Single `"bypass_lora"` injection key. A shared `BypassInjectionManager` is stored as a model attachment via `get_attachment("bypass_lora_manager")` / `set_attachments()`. Subsequent LoRA loaders add adapters to the existing manager instead of creating new injection sets. `create_injections()` rebuilds all hooks fresh each time.

### `comfy/weight_adapter/bypass.py` — Per-Layer Adapter Offloading

**Problem:** With `--normalvram`, adapter weights permanently on GPU waste VRAM. On a 24GB card running Wan 2.2 I2V 14B INT8, every MB counts.

**Fix:** Adapter weights move to GPU at the start of each `_bypass_forward()` call, then back to CPU in a `finally` block. Only one layer's LoRA weights are on GPU at any time, mirroring `--normalvram` behavior for base model weights.

```python
def _bypass_forward(self, x, *args, **kwargs):
    self._move_adapter_weights_to_device(x.device, x.dtype)
    try:
        base_out = self.original_forward(x, *args, **kwargs)
        h_out = self.adapter.h(x, base_out)
        return self.adapter.g(base_out + h_out)
    finally:
        self._move_adapter_weights_to_device(torch.device("cpu"))
```

## Use Case

This fork is designed for:
- **INT8 quantized models** (W8A8) where bypass LoRA avoids re-quantization artifacts
- **Multiple stacked LoRAs** (distilled/turbo + style/NSFW) without infinite recursion
- **Low VRAM setups** (`--normalvram`) where per-layer offloading is critical

## Staying Up to Date

This fork tracks upstream `comfyanonymous/ComfyUI` master. To update:

```bash
git fetch origin
git merge origin/master
```

The two modified files may conflict on merge — resolve by reapplying the changes described above.
