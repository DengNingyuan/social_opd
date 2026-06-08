from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data import parse_paths
from experiment_config import default_run_name, load_named_config, write_metadata
from inference import run_inference, write_jsonl
from metrics import compute_measurement_metrics
from train_opd import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run measurement inference and compute ECE/Brier/MH.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--loss_version", type=str, default="lossv1")
    parser.add_argument("--loss_config", type=Path, default=Path("configs/losses.json"))
    parser.add_argument("--teacher_version", type=str, default="teacherv1")
    parser.add_argument("--teacher_config", type=Path, default=Path("configs/teachers.json"))
    parser.add_argument("--test_files", type=str, default=None, help="Comma-separated preprocessed test JSONL files.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    dataset = load_named_config(args.dataset_config, args.dataset_name)
    loss = load_named_config(args.loss_config, args.loss_version)
    teacher = load_named_config(args.teacher_config, args.teacher_version)
    output_dir = args.output_dir or Path("outputs/infer") / default_run_name(
        "infer",
        args.dataset_name,
        args.loss_version,
        args.teacher_version,
        args.model_path,
    )
    device = choose_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    infer_args = SimpleNamespace(
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        max_test_samples=args.max_test_samples,
    )
    test_files = args.test_files or str(Path("data/processed") / args.dataset_name / "test.jsonl")
    predictions = run_inference(model, tokenizer, parse_paths(test_files), infer_args, device, expected_dataset=args.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    metrics = compute_measurement_metrics(predictions, tolerance=args.tolerance, bins=args.ece_bins)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metadata(
        output_dir / "run_metadata.json",
        {
            "mode": "direct_inference",
            "model_path": args.model_path,
            "output_dir": str(output_dir),
            "dataset": dataset,
            "test_files": test_files,
            "loss": loss,
            "teacher": teacher,
            "metrics_file": "metrics.json",
            "predictions_file": "predictions.jsonl",
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
