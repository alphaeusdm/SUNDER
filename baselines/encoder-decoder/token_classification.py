import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from metrics import *
from utils import *
from datasets import Dataset as HFDataset
from langchain_core.prompts import PromptTemplate
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)


MODEL_NAME   = "google/flan-t5-xl" 
MAX_SRC_LEN  = 500
MAX_TGT_LEN  = 500
BATCH_SIZE   = 2
LR           = 1e-4
MAX_STEPS    = 1000
OUTPUT_DIR   = "models/{task}/flan/"

def format_example(task, example):
    if "NER" in task:
        task="NER"
    format_functions = {
        'NER': format_function_ner,
        'DEF': format_function_def,
        'ABSA': format_function_absa,
    }
    source, target = format_functions[task](example)
    if not target:
        target = "none"
    return source, target


def build_instruction_dataset(task, hf_dataset):
    examples = {"input": [], "target": []}
    for i, ex in enumerate(hf_dataset):
        source, target = format_example(task, ex)
        examples["input"].append(source)
        examples["target"].append(target)
    return examples


def load_data(task_name="NER", split="test"):
    data = pd.read_csv(f'data/{task_name}/{split}.csv', sep='\t')
    if split != "test":
        data = HFDataset.from_pandas(data)
    return data


def make_tokenize_fn(tokenizer):
    def tokenize(examples):
        # Encoder input
        model_inputs = tokenizer(
            examples["input"],
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding=False,   # DataCollatorForSeq2Seq handles padding dynamically
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=MAX_TGT_LEN,
            truncation=True,
            padding=False,
        )
        labels["input_ids"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenize


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        predictions = np.where(
            predictions < 0, tokenizer.pad_token_id, predictions
        )

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        correct = pred_total = gold_total = 0
        for pred, gold in zip(decoded_preds, decoded_labels):
            pred_ents = set(e.strip() for e in pred.split(",") if e.strip())
            gold_ents = set(e.strip() for e in gold.split(",") if e.strip())
            correct += len(pred_ents & gold_ents)
            pred_total += len(pred_ents)
            gold_total += len(gold_ents)

        precision = correct / pred_total if pred_total else 0.0
        recall = correct / gold_total if gold_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        return {"precision": precision, "recall": recall, "entity_f1": f1}
    return compute_metrics


def train(task):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dict = build_instruction_dataset(task, load_data(task, "train"))
    val_dict = build_instruction_dataset(task, load_data(task, "dev"))

    train_ds = HFDataset.from_dict(train_dict)
    val_ds = HFDataset.from_dict(val_dict)

    # Tokenize
    tokenize_fn = make_tokenize_fn(tokenizer)
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])
    val_tok = val_ds.map(tokenize_fn,   batched=True, remove_columns=["input", "target"])

    # DataCollator handles dynamic padding for both input_ids and labels
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR.format(task=task),
        eval_steps=100,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        # per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=LR,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        save_strategy="steps",
        predict_with_generate=True,
        generation_max_length=MAX_TGT_LEN,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        save_steps=100,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR.format(task=task))
    tokenizer.save_pretrained(OUTPUT_DIR.format(task=task))
    print(f"Model saved to {OUTPUT_DIR.format(task=task)}")


def load_prompt(name):
    if "NER" in name:
        name="NER"
    with open(f"prompts/{name.lower()}.json", 'r') as fp:
        prompt = json.load(fp)
    
    return prompt['prompt']

def predict(prompt, model, tokenizer):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_SRC_LEN,
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TGT_LEN,
        num_beams=4,
        early_stopping=True,
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = output.strip()
    output = re.sub(r"\n\n",r"\n",output)
    output = output.strip().split(' | ')
    
    return output

def test(task, id2label):
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR.format(task=task))
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR.format(task=task))
    model.eval()

    test_data = load_data(task, "test")
    columns = list(test_data.columns)[1:]
    instruction_prompt = load_prompt(task)
    template = "{instruction}\n\nInput:\nText: {sentence}\n"
    prompt = PromptTemplate(template=template, input_variables=["instruction","sentence"])

    gold_arguments = list()
    predicted_arguments = list()
    for ind, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        if "NER" in task:
            bin_argument = get_gold_labels("NER", row, columns)
        else:
            bin_argument = get_gold_labels(task, row, columns)
        gold_arguments.append(bin_argument)
        sentence = row['text']
        generate_prompt = prompt.format(instruction=instruction_prompt, sentence=sentence)
        output = predict(generate_prompt, model, tokenizer)
        if "NER" in task:
            bin_argument = get_spans("NER", row, output)
        else:
            bin_argument = get_spans(task, row, output)
        predicted_arguments.append(bin_argument)

        if len(gold_arguments[-1]) != len(predicted_arguments[-1]):
            print(ind)
            gold_arguments = gold_arguments[:-1]
            predicted_arguments = predicted_arguments[:-1]
        
    predicted_arguments_lb = [[id2label[p] for p in pred] for pred in predicted_arguments]
    gold_arguments_lb = [[id2label[l] for l in label] for label in gold_arguments]
    argument_scores = get_metrics(gold_arguments, predicted_arguments)
    print("Argument Token F1: {}, Argument Span F1: {}".format(argument_scores[0], argument_scores[1]))
    print("SeqEval F1: ",round(get_ner_metrics(gold_arguments_lb, predicted_arguments_lb)['overall_f1'], 3))
    metrics = get_ner_metrics(gold_arguments_lb, predicted_arguments_lb)
    for k, v in metrics.items():
        if isinstance(v,dict):
            print(f"{k}: {round(v['f1'],3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="NER")
    args = parser.parse_args()

    if args.task == "ABSA":
        dict_idx2lbl = {0: 'O', 1: 'B-Aspect', 2: 'I-Aspect', 3: 'B-Opinion', 4: 'I-Opinion'}
    elif "NER" in args.task:
        dict_idx2lbl = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    elif args.task == "DEF":
        dict_idx2lbl = {
            0: 'O', 1: 'B-Alias-Term', 2: 'I-Alias-Term', 3: 'B-Alias-Term-frag', 4: 'I-Alias-Term-frag', 5: 'B-Definition', 
            6: 'I-Definition', 7: 'B-Definition-frag', 8: 'I-Definition-frag', 9: 'B-Ordered-Definition', 10: 'I-Ordered-Definition', 
            11: 'B-Ordered-Term', 12: 'I-Ordered-Term', 13: 'B-Qualifier', 14: 'I-Qualifier', 15: 'B-Referential-Definition', 
            16: 'I-Referential-Definition', 17: 'B-Referential-Term', 18: 'I-Referential-Term', 19: 'B-Secondary-Definition', 
            20: 'I-Secondary-Definition', 21: 'B-Term', 22: 'I-Term', 23: 'B-Term-frag', 24: 'I-Term-frag'
        }
    else:
        raise ValueError("Invalid task")

    train(args.task)
    test(args.task, dict_idx2lbl) 
    