"""Continued pretraining of ModernBERT on MIMIC-IV with the HF Trainer.

    python -m trialcode.pretrain --mimic-root /data/mimic --variant all
"""
import argparse
import os

from transformers import (AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, set_seed)

from trialcode import tracking  # noqa: F401  (loads .env for W&B)
from trialcode.data.collator import CodeAwareMLMCollator
from trialcode.data.mimic import load_mimic, split_by_subject
from trialcode.data.tokenize import make_tokenize_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mimic-root", required=True)
    p.add_argument("--variant", default="all", choices=["all", "structured", "codes"])
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-code-tokens", type=int, default=512)
    p.add_argument("--code-prob", type=float, default=0.5)
    p.add_argument("--max-train-examples", type=int, default=None, help="debug subsampling")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()

    set_seed(a.seed)
    run_name = f"cpt-{a.variant}-{a.model.split('/')[-1]}"
    out = a.output_dir or os.path.join("outputs", run_name)

    tok = AutoTokenizer.from_pretrained(a.model)
    splits = split_by_subject(load_mimic(a.mimic_root, a.variant))
    if a.max_train_examples:
        splits["train"] = splits["train"].select(range(min(a.max_train_examples, len(splits["train"]))))
    tokenize = make_tokenize_fn(tok, a.max_length, a.max_code_tokens)
    splits = {k: v.map(tokenize, batched=True, remove_columns=v.column_names, desc=f"tokenize {k}")
              for k, v in splits.items()}

    model = AutoModelForMaskedLM.from_pretrained(a.model)
    args = TrainingArguments(
        output_dir=out,
        run_name=run_name,
        report_to="none" if a.no_wandb else "wandb",
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=0.05,  # <1 = fraction of total steps,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        bf16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        logging_steps=25,
        remove_unused_columns=False,  # collator needs code_group/code_part
        dataloader_num_workers=4,
        seed=a.seed,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=splits["train"], eval_dataset=splits["validation"],
        data_collator=CodeAwareMLMCollator(tok, code_prob=a.code_prob),
    )
    trainer.train()
    trainer.save_model(os.path.join(out, "final"))


if __name__ == "__main__":
    main()
