#!/usr/bin/env python3
"""
export_onnx.py - Export Kokoro-82M PyTorch Checkpoint to INT8 Quantized ONNX Model.
Exposes input_ids, voicepack style tensor, and speed modifier as dynamic ONNX inputs.
Applies Dynamic INT8 Quantization (QUInt8/QInt8) to shrink model footprint to ~80 MB.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from onnxruntime.quantization import quantize_dynamic, QuantType

DEFAULT_CONDITIONING_NODES = (
    "style_projection",
    "duration_predictor",
    "emotion_embedding",
)


class KokoroONNXWrapper(nn.Module):
    """
    Export wrapper for Kokoro-82M acoustic decoder.
    Exposes input_ids, voicepack style tensor, and speed modifier as dynamic ONNX inputs.
    """
    def __init__(self, kmodel):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids: torch.Tensor, style: torch.Tensor, speed: torch.Tensor):
        # input_ids: int64 [1, seq_len]
        # style: float32 [510, 1, 256]
        # speed: float32 [1]
        
        # Forward pass exposing speed into predictor duration scaling
        if hasattr(self.kmodel, "forward_with_tokens"):
            audio, durations = self.kmodel.forward_with_tokens(
                input_ids=input_ids,
                ref_s=style,
                speed=speed
            )
            return audio, durations
        else:
            return self.kmodel(input_ids, style, speed)


def quantize_with_conditioning_exclusions(
    fp32_onnx: Path,
    int8_onnx: Path,
    conditioning_nodes: tuple[str, ...] = DEFAULT_CONDITIONING_NODES,
) -> None:
    """Quantize generator/vocoder weights while retaining conditioning heads in float.

    Node names are model-export specific. Pass the names emitted by the trained
    model's ONNX graph through ``--conditioning_nodes`` if they differ from the
    project defaults.
    """
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=list(conditioning_nodes),
    )


def export_and_quantize(
    model_pth: Path,
    output_dir: Path,
    conditioning_nodes: tuple[str, ...] = DEFAULT_CONDITIONING_NODES,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = output_dir / "kokoro-expressive.onnx"
    int8_onnx = output_dir / "kokoro-expressive.int8.onnx"

    # 1. Load PyTorch model
    print(f"Loading checkpoint from {model_pth}...")
    kmodel = torch.load(model_pth, map_location="cpu")
    if hasattr(kmodel, "eval"):
        kmodel.eval()

    wrapper = KokoroONNXWrapper(kmodel)

    # 2. Define Dummy Inputs
    dummy_input_ids = torch.randint(1, 100, (1, 50), dtype=torch.int64)
    dummy_style = torch.randn(510, 1, 256, dtype=torch.float32)  # Standard 3D voicepack tensor
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    # 3. Export to FP32 ONNX
    print("Exporting FP32 ONNX graph...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_style, dummy_speed),
        str(fp32_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["audio", "durations"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "style": {0: "style_dim"},
            "audio": {1: "audio_len"},
            "durations": {1: "num_tokens"},
        },
    )

    # 4. Perform INT8 Dynamic Quantization
    print("Applying INT8 Dynamic Quantization...")
    quantize_with_conditioning_exclusions(fp32_onnx, int8_onnx, conditioning_nodes)

    print(f"ONNX Quantization Complete!\n  - FP32 Model: {fp32_onnx}\n  - INT8 Quantized Model: {int8_onnx}")


def quantize_existing_onnx(
    fp32_onnx: Path,
    int8_onnx: Path,
    conditioning_nodes: tuple[str, ...] = DEFAULT_CONDITIONING_NODES,
):
    """Quantize an existing graph without converting style/duration heads to INT8."""
    print(f"Quantizing {fp32_onnx} -> {int8_onnx}...")
    quantize_with_conditioning_exclusions(fp32_onnx, int8_onnx, conditioning_nodes)
    print("Quantization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Kokoro-82M to INT8 ONNX.")
    parser.add_argument("--model", type=Path, help="Input Stage 2 .pth checkpoint")
    parser.add_argument("--onnx", type=Path, help="Existing FP32 ONNX to quantize directly")
    parser.add_argument("--out_dir", type=Path, default=Path("models_onnx"))
    parser.add_argument(
        "--conditioning_nodes",
        default=",".join(DEFAULT_CONDITIONING_NODES),
        help="Comma-separated ONNX node names kept in FP16/FP32 during quantization",
    )
    args = parser.parse_args()
    conditioning_nodes = tuple(node.strip() for node in args.conditioning_nodes.split(",") if node.strip())

    if args.model:
        export_and_quantize(args.model, args.out_dir, conditioning_nodes)
    elif args.onnx:
        out = args.out_dir / "kokoro-expressive.int8.onnx"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        quantize_existing_onnx(args.onnx, out, conditioning_nodes)
    else:
        print("Please provide --model (PyTorch .pth) or --onnx (FP32 .onnx).")
