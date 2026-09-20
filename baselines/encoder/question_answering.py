import os
import torch
import argparse
import numpy as np
import pandas as pd
import collections
from metrics import *
from peft import LoraConfig, PeftModel, get_peft_model, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

import warnings
warnings.filterwarnings("ignore")


class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_seq_len, doc_stride=128, test=False):
        super().__init__()
        self.test = test

        self.example_ids = data["id"].tolist()
        questions = data["question"].astype(str).tolist()
        contexts  = data["context"].astype(str).tolist()

        # Filter out questions that are too long to leave room for any context
        num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        valid_indices = []
        for i, q in enumerate(questions):
            q_len = len(tokenizer(q, add_special_tokens=False)["input_ids"])
            if q_len < max_seq_len - num_special_tokens - 1:
                valid_indices.append(i)

        if len(valid_indices) < len(questions):
            print(f"Filtered {len(questions) - len(valid_indices)} examples with questions too long")

        questions        = [questions[i]        for i in valid_indices]
        contexts         = [contexts[i]         for i in valid_indices]
        self.example_ids = [self.example_ids[i] for i in valid_indices]
        if not test:
            data = data.iloc[valid_indices].reset_index(drop=True)

        # Step 1: tokenize without padding — keeps BatchEncoding intact
        encodings = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        # Step 2: extract everything that needs BatchEncoding BEFORE padding
        # sample_mapping may index into filtered list, remap accordingly
        raw_sample_mapping   = encodings.pop("overflow_to_sample_mapping")
        raw_offset_mapping   = encodings.pop("offset_mapping")

        # sequence_ids() must be called before pad() destroys BatchEncoding
        # Pad sequence_ids to max_seq_len with None for padding positions
        self.sequence_ids_list = [
            encodings.sequence_ids(i) + [None] * (max_seq_len - len(encodings.sequence_ids(i)))
            for i in range(len(encodings["input_ids"]))
        ]

        self.sample_mapping = raw_sample_mapping

        # Pad offset_mapping to max_seq_len with (0, 0) for padding positions
        self.offset_mapping = [
            list(om) + [(0, 0)] * (max_seq_len - len(om))
            for om in raw_offset_mapping
        ]

        # Step 3: pad — only input_ids, attention_mask, token_type_ids remain
        padded = tokenizer.pad(
            encodings,
            padding="max_length",
            max_length=max_seq_len,
            return_tensors="pt",
        )
        self.input_ids      = padded["input_ids"]
        self.attention_mask = padded["attention_mask"]
        self.token_type_ids = padded.get("token_type_ids", None)

        if not test:
            answer_texts  = data["answer_text"].astype(str).tolist()
            answer_starts = data["answer_start"].tolist()
            self._set_token_labels(answer_texts, answer_starts)

    def _set_token_labels(self, answer_texts, answer_starts):
        self.start_positions = []
        self.end_positions   = []

        for i, offsets in enumerate(self.offset_mapping):
            sample_idx   = self.sample_mapping[i]
            answer_text  = answer_texts[sample_idx]
            answer_start = int(answer_starts[sample_idx])
            answer_end   = answer_start + len(answer_text)

            sequence_ids = self.sequence_ids_list[i]

            if 1 not in sequence_ids:
                self.start_positions.append(0)
                self.end_positions.append(0)
                continue

            context_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
            context_end   = max(j for j, s in enumerate(sequence_ids) if s == 1)

            ctx_char_start = offsets[context_start][0]
            ctx_char_end   = offsets[context_end][1]

            # Remap answer_start/end relative to what the tokenizer actually sees
            # by finding overlap between the answer span and the token offsets
            if ctx_char_end <= answer_start or ctx_char_start >= answer_end:
                # Answer not in this window
                self.start_positions.append(0)
                self.end_positions.append(0)
                continue

            # Find start token: last token whose start offset <= answer_start
            token_start = context_start
            for j in range(context_start, context_end + 1):
                if offsets[j][0] <= answer_start and offsets[j][1] > answer_start:
                    token_start = j
                # else:
                    break

            # Find end token: first token whose end offset >= answer_end
            token_end = token_start
            for j in range(token_start, context_start + 1):
                if offsets[j][1] >= answer_end:
                    token_end = j
                # else:
                    break

            self.start_positions.append(token_start)
            self.end_positions.append(token_end)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if not self.test:
            item["start_positions"] = torch.tensor(self.start_positions[idx])
            item["end_positions"]   = torch.tensor(self.end_positions[idx])
        return item

    def __len__(self):
        return len(self.input_ids)


def normalize_answer(s):
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def compute_exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common      = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_common  = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def postprocess_predictions(dataset, raw_predictions, n_best=20, max_answer_len=30):
    start_logits, end_logits = raw_predictions

    # Collect candidates per example across all sliding windows
    all_candidates = collections.defaultdict(list)

    for i in range(len(dataset)):
        sample_idx   = dataset.sample_mapping[i]
        example_id   = dataset.example_ids[sample_idx]
        offsets      = dataset.offset_mapping[i]
        sequence_ids = dataset.sequence_ids_list[i]

        if 1 not in sequence_ids:
            continue

        context_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
        context_end   = max(j for j, s in enumerate(sequence_ids) if s == 1)

        for s_idx in np.argsort(start_logits[i])[::-1][:n_best]:
            for e_idx in np.argsort(end_logits[i])[::-1][:n_best]:
                if sequence_ids[s_idx] != 1 or sequence_ids[e_idx] != 1:
                    continue
                if e_idx < s_idx or (e_idx - s_idx + 1) > max_answer_len:
                    continue
                if s_idx < context_start or e_idx > context_end:
                    continue
                all_candidates[example_id].append((
                    float(start_logits[i][s_idx] + end_logits[i][e_idx]),
                    offsets[s_idx][0],
                    offsets[e_idx][1],
                ))

    # Pick the best scoring span per example across all windows
    predictions = {}
    for example_id, candidates in all_candidates.items():
        predictions[example_id] = max(candidates, key=lambda x: x[0])

    # Fallback for examples with no valid candidates
    for example_id in dataset.example_ids:
        if example_id not in predictions:
            predictions[example_id] = ("", 0, 0)

    return predictions


def get_qa_data(csv_path, data_size=None):
    data = (
        pd.read_csv(csv_path, sep="\t").sample(data_size, random_state=2707)
        if data_size
        else pd.read_csv(csv_path, sep="\t")
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   type=str,   default="data")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--task",        type=str,   default="QA")
    parser.add_argument("--model_name",  type=str,   default="llama")
    parser.add_argument("--model_size",  type=int,   default=8)
    parser.add_argument("--data_size",   type=int,   default=None)
    parser.add_argument("--max_seq_len", type=int,   default=512)
    parser.add_argument("--doc_stride",  type=int,   default=128)
    parser.add_argument("--bs",          type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--max_steps",   type=int,   default=1000)
    args = parser.parse_args()

    assert args.doc_stride < args.max_seq_len, \
        f"doc_stride ({args.doc_stride}) must be less than max_seq_len ({args.max_seq_len})"


    if args.model_name == "roberta":
        hf_model_name = "FacebookAI/roberta-large"
    else:
        hf_model_name = "google-bert/bert-large-uncased"
    output_dir = f"models/{args.task}/{args.model_name}-{args.model_size}B-question-answering"

    print(f"Output dir : {output_dir}")
    print(f"Loading    : {hf_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)

    if args.train:
        model = AutoModelForQuestionAnswering.from_pretrained(
            hf_model_name,
            ignore_mismatched_sizes=True,
            device_map="auto",
            dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            output_dir,
            ignore_mismatched_sizes=True,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        model.eval()

    if args.train:
        train_data = get_qa_data(f"{args.data_path}/{args.task}/train.csv", args.data_size)
        dev_data   = get_qa_data(f"{args.data_path}/{args.task}/dev.csv")

        trainset = QADataset(train_data, tokenizer, args.max_seq_len, args.doc_stride)
        devset   = QADataset(dev_data,   tokenizer, args.max_seq_len, args.doc_stride)

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_steps=100,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=args.lr,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_32bit",
            eval_strategy="steps",
            save_strategy="steps",
            fp16=False, bf16=True,
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_steps=100, logging_steps=100, save_total_limit=2,
            report_to="none",
        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=trainset, eval_dataset=devset,
            processing_class=tokenizer,
            data_collator=default_data_collator,
        )
        trainer.train()
        trainer.save_model(output_dir)

    else:
        test_data = get_qa_data(f"{args.data_path}/{args.task}/test.csv")
        testset   = QADataset(test_data, tokenizer, args.max_seq_len, args.doc_stride, test=True)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir, per_device_eval_batch_size=args.bs),
            data_collator=default_data_collator,
        )
        raw_preds  = trainer.predict(testset)
        pred_spans = postprocess_predictions(testset, raw_preds.predictions)

        gold_answers = dict(zip(test_data["id"].tolist(), test_data["answer_text"].tolist()))
        contexts_map = dict(zip(test_data["id"].tolist(), test_data["context"].tolist()))

        gold_outputs, predicted_outputs = [], []
        for ex_id, (score, start_char, end_char) in pred_spans.items():
            context   = contexts_map[ex_id]
            pred_text = context[start_char:end_char].strip()
            gold_text = str(gold_answers.get(ex_id, "")).strip()
            gold_outputs.append(gold_text)
            predicted_outputs.append(pred_text)

        output_scores = get_mhqa_metrics(gold_outputs, predicted_outputs)
        print(f"EM: {output_scores['em']}, F1: {output_scores['f1']}")