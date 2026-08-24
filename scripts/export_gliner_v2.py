#!/usr/bin/env python3
"""Export vicgalle/gliner-small-pii to ONNX INT8 — the weights PIIMasker loads.

Two steps, both public so the published model can be reproduced rather than
trusted: this script turns the upstream PyTorch model into a quantized ONNX
graph, and package-model.sh turns that into the tarball ModelPin names.

The upstream revision is PINNED. Pinning the Python toolchain (requirements.txt)
proves nothing while the weights being exported can move underneath it.

Usage:  python3 export_gliner_v2.py --out DIR
        pip install -r requirements.txt
"""

import argparse
import os
import json
import shutil
import tempfile

import onnx
from onnx import helper, TensorProto
from gliner import GLiNER

MODEL_ID = "vicgalle/gliner-small-pii"
# Apache-2.0, itself fine-tuned from gliner-community/gliner_small-v2.5. Pinned to
# a commit rather than a branch so this export names exactly one set of weights.
MODEL_REVISION = "aaafd3bbdf84fe6dadda12f6320b40a5bb920193"


def patch_span_mask_to_int64(model_path, output_path):
    """Change span_mask input from Bool to Int64, inserting a Cast node."""
    model = onnx.load(model_path)

    for inp in model.graph.input:
        if inp.name == "span_mask":
            old_type = inp.type.tensor_type.elem_type
            if old_type == TensorProto.BOOL:
                inp.name = "span_mask_int64"
                inp.type.tensor_type.elem_type = TensorProto.INT64
                cast_node = helper.make_node(
                    "Cast", inputs=["span_mask_int64"], outputs=["span_mask"],
                    to=TensorProto.BOOL,
                )
                model.graph.node.insert(0, cast_node)
                print("  Patched span_mask: Bool → Int64 (with Cast node)")
            else:
                print("  span_mask already Int64, no patch needed")
            break

    onnx.save(model, output_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", required=True,
        help="directory to write model.onnx, the tokenizer files and gliner_config.json into")
    ap.add_argument(
        "--revision", default=MODEL_REVISION,
        help="upstream commit to export (default: the pinned one)")
    args = ap.parse_args()
    output_dir = args.out

    print(f"Loading {MODEL_ID} @ {args.revision}...")
    model = GLiNER.from_pretrained(MODEL_ID, revision=args.revision)
    model.eval()

    print(f"Span mode: {model.config.span_mode}")
    print(f"Max width: {model.config.max_width}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Class token index: {model.config.class_token_index}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\nExporting to ONNX...")
        paths = model.export_to_onnx(tmpdir, quantize=False)
        onnx_path = paths.get("onnx_path")
        print(f"  Original: {os.path.getsize(onnx_path) / (1024 * 1024):.1f} MB")

        # Check if span_mask needs patching
        onnx_model = onnx.load(onnx_path)
        needs_patch = False
        for inp in onnx_model.graph.input:
            if inp.name == "span_mask" and inp.type.tensor_type.elem_type == TensorProto.BOOL:
                needs_patch = True
                break

        if needs_patch:
            patched_path = os.path.join(tmpdir, "model_patched.onnx")
            patch_span_mask_to_int64(onnx_path, patched_path)
            quant_input = patched_path
        else:
            quant_input = onnx_path

        # Quantize
        print("Quantizing to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = os.path.join(tmpdir, "model_quantized.onnx")
        quantize_dynamic(quant_input, quantized_path, weight_type=QuantType.QInt8)
        print(f"  Quantized: {os.path.getsize(quantized_path) / (1024 * 1024):.1f} MB")

        # Verify final I/O
        final = onnx.load(quantized_path)
        print("\nFinal ONNX inputs:")
        for inp in final.graph.input:
            shape = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
            dtype = inp.type.tensor_type.elem_type
            print(f"  {inp.name}: shape={shape} dtype={dtype}")
        print("Final ONNX outputs:")
        for out in final.graph.output:
            shape = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
            dtype = out.type.tensor_type.elem_type
            print(f"  {out.name}: shape={shape} dtype={dtype}")

        # Copy to output
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(quantized_path, os.path.join(output_dir, "model.onnx"))
        print(f"\nCopied model.onnx to {output_dir}")

        # Copy tokenizer files
        for f in os.listdir(tmpdir):
            if f.startswith("tokenizer"):
                shutil.copy2(os.path.join(tmpdir, f), os.path.join(output_dir, f))
                print(f"  Copied {f}")

        # Save gliner config
        config_dict = {}
        for k, v in model.config.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                config_dict[k] = v
            except (TypeError, ValueError):
                config_dict[k] = str(v)
        with open(os.path.join(output_dir, "gliner_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        print("  Saved gliner_config.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
