"""
ONNX Export and Quantization for fine-tuned Kokoro TTS models.

Features:
1. PyTorch to ONNX graph export with dynamic axes.
2. Dynamic INT8 quantization for fast, efficient CPU inference.
"""
import os
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(input_onnx_path: str, output_onnx_path: str) -> str:
    """
    Quantize full precision FP32 ONNX model to INT8.
    Reduces model size from ~320MB to ~80MB and significantly speeds up CPU synthesis.
    """
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"Source model not found: {input_onnx_path}")
        
    print(f"Quantizing {input_onnx_path} -> {output_onnx_path} (INT8)...")
    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        weight_type=QuantType.QInt8,
    )
    
    orig_mb = os.path.getsize(input_onnx_path) / (1024 * 1024)
    quant_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Quantization complete! {orig_mb:.1f} MB -> {quant_mb:.1f} MB ({(orig_mb/quant_mb):.1f}x compression)")
    return output_onnx_path


def export_pytorch_to_onnx(model, output_path: str):
    """
    Export a PyTorch KModel to ONNX format.
    Run on GPU / Colab where PyTorch model is instantiated.
    """
    import torch
    model.eval()
    
    dummy_input_ids = torch.zeros(1, 32, dtype=torch.long)
    dummy_style = torch.zeros(1, 256, dtype=torch.float32)
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_style, dummy_speed),
        output_path,
        input_names=["input_ids", "style", "speed"],
        output_names=["audio"],
        dynamic_axes={
            "input_ids": {1: "input_ids_len"},
            "audio": {1: "audio_len"},
        },
        opset_version=17,
    )
    print(f"Exported PyTorch model to ONNX: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize or export Kokoro ONNX model.")
    parser.add_argument("--input", type=str, required=True, help="Input FP32 ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Output quantized INT8 ONNX model")
    args = parser.parse_args()
    
    quantize_onnx_model(args.input, args.output)
