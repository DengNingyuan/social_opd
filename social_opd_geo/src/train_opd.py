from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import OpdCollator, ProcessedJsonlDataset, parse_paths
from experiment_config import default_run_name, load_named_config, write_metadata
from inference import run_inference, write_jsonl
from losses import self_opd_reverse_kl
from metrics import compute_measurement_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Self-OPD and evaluate on a measurement test set.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="formality")
    parser.add_argument("--dataset_config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--loss_version", type=str, default="lossv1")
    parser.add_argument("--loss_config", type=Path, default=Path("configs/losses.json"))
    parser.add_argument("--teacher_version", type=str, default="teacherv1")
    parser.add_argument("--teacher_config", type=Path, default=Path("configs/teachers.json"))
    parser.add_argument("--train_files", type=str, default=None, help="Comma-separated preprocessed train JSONL files.")
    parser.add_argument("--test_files", type=str, default=None, help="Comma-separated preprocessed test JSONL files.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--teacher_device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    teacher_device = choose_teacher_device(args.teacher_device, device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dataset = load_named_config(args.dataset_config, args.dataset_name)
    loss_config = load_named_config(args.loss_config, args.loss_version)
    teacher_config = load_named_config(args.teacher_config, args.teacher_version)
    processed_dir = Path("data/processed") / args.dataset_name
    train_files = args.train_files or str(processed_dir / "train.jsonl")
    test_files = args.test_files or str(processed_dir / "test.jsonl")
    output_dir = args.output_dir or Path("outputs/train") / default_run_name(
        "train",
        args.dataset_name,
        args.loss_version,
        args.teacher_version,
        args.model_name,
    )

    collator = OpdCollator(tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        ProcessedJsonlDataset(parse_paths(train_files), max_samples=args.max_train_samples, expected_dataset=args.dataset_name),
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(teacher_device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    model.config.pad_token_id = tokenizer.pad_token_id
    teacher.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    total_steps = args.max_steps or math.ceil(len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    with tqdm(total=total_steps, desc="train", unit="step") as progress:
        for epoch in range(args.num_train_epochs):
            accumulated = 0
            epoch_loader = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1}/{args.num_train_epochs}",
                unit="batch",
                leave=False,
            )
            for batch_idx, batch in enumerate(epoch_loader):
                loss, stats = compute_batch_loss(
                    model,
                    teacher,
                    move_training_batch(batch, device, teacher_device),
                    args.lambda_kl,
                    device,
                )
                (loss / args.gradient_accumulation_steps).backward()
                accumulated += 1
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    accumulated = 0
                    progress.update(1)
                    progress.set_postfix(loss=f"{stats['loss']:.4f}", reverse_kl=f"{stats['reverse_kl']:.4f}")
                    tqdm.write(f"step={step} loss={stats['loss']:.4f} reverse_kl={stats['reverse_kl']:.4f}")
                    if args.max_steps is not None and step >= args.max_steps:
                        break
            if accumulated > 0 and (args.max_steps is None or step < args.max_steps):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accumulated = 0
                progress.update(1)
                progress.set_postfix(loss=f"{stats['loss']:.4f}", reverse_kl=f"{stats['reverse_kl']:.4f}")
                tqdm.write(f"step={step} loss={stats['loss']:.4f} reverse_kl={stats['reverse_kl']:.4f}")
            if args.max_steps is not None and step >= args.max_steps:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    predictions = run_inference(model, tokenizer, parse_paths(test_files), args, device, expected_dataset=args.dataset_name)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    metrics = compute_measurement_metrics(predictions, tolerance=args.tolerance, bins=args.ece_bins)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metadata(
        output_dir / "run_metadata.json",
        {
            "mode": "train_then_infer",
            "model_name": args.model_name,
            "output_dir": str(output_dir),
            "saved_model_dir": "model",
            "dataset": dataset,
            "train_files": train_files,
            "test_files": test_files,
            "loss": loss_config,
            "teacher": teacher_config,
            "metrics_file": "metrics.json",
            "predictions_file": "predictions.jsonl",
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def compute_batch_loss(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    batch: dict[str, Any],
    lambda_kl: float,
    student_device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_chosen_out = model(**batch["student_chosen"])
    teacher_inputs = {
        key: value for key, value in batch["teacher_chosen"].items()
        if key != "labels"
    }
    with torch.no_grad():
        teacher_chosen_out = teacher(**teacher_inputs)
    reverse_kl = self_opd_reverse_kl(
        student_chosen_out.logits,
        teacher_chosen_out.logits.to(student_device),
        batch["student_chosen"]["labels"],
        batch["teacher_chosen"]["labels"].to(student_device),
    )
    loss = lambda_kl * reverse_kl
    return loss, {
        "loss": float(loss.detach().cpu()),
        "reverse_kl": float(reverse_kl.detach().cpu()),
    }


def move_training_batch(batch: dict[str, Any], student_device: torch.device, teacher_device: torch.device) -> dict[str, Any]:
    return {
        "student_chosen": {key: value.to(student_device) for key, value in batch["student_chosen"].items()},
        "teacher_chosen": {key: value.to(teacher_device) for key, value in batch["teacher_chosen"].items()},
        "texts": batch["texts"],
        "ground_truth": batch["ground_truth"].to(student_device),
    }


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_teacher_device(value: str, student_device: torch.device) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if student_device.type == "cuda" and torch.cuda.device_count() >= 2:
        return torch.device("cuda:1")
    return student_device


if __name__ == "__main__":
    main()
