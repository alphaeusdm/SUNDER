'''
part of the code reused from x-claim repository: https://github.com/mbzuai-nlp/x-claim
'''
import os
import torch
import torch.distributed as dist
import argparse
import stanza
import numpy as np
import pandas as pd
from metrics import *
from utils import *
from datasets import Dataset, concatenate_datasets, IterableDataset
from trl import SFTTrainer, SFTConfig 
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.data_utils import truncate_dataset, pack_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from modeling import (
    modeling_llama,
    modeling_mistral,
    modeling_qwen,
    modeling_qwen3,
)
from typing import Any, Callable, Optional, TypeVar, Union
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from accelerate import PartialState, logging
from trl.trainer.utils import pad

import warnings
warnings.filterwarnings("ignore")


class myDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_seq_len, dict_lbl2idx, label_all_tokens=True, test=False):
        super(myDataset, self).__init__()
        self.test = test
        self.examples = examples
        self.seqs = examples[0]
        # self.flag = False

        # tokenize each word manually since there are weird words preseq like '\ufeff' or '' which tokenize into nothing
        seqs_tokenized = {
            'input_ids': [],
            'attention_mask': [],
            'word_ids': []
        }
        model_name = tokenizer.name_or_path.lower()
        if "qwen" in model_name:
            eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
        else:
            bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id

        for (idx, seq) in enumerate(self.seqs):
            input_ids, attention_mask, word_ids = [], [], []
            for idx, word in enumerate(seq):
                token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
                if len(token_ids) == 0:
                    token_ids = [pad_id]
                if (len(input_ids) + len(token_ids) > max_seq_len): # not test and 
                    break

                input_ids += token_ids
                attention_mask += ([1] * len(token_ids))
                word_ids += ([idx] * len(token_ids))

            pad_len = max_seq_len - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            word_ids = word_ids + [None] * pad_len

            assert len(input_ids) == len(attention_mask) and len(word_ids) == len(attention_mask)
            
            seqs_tokenized['input_ids'].append(input_ids)
            seqs_tokenized['attention_mask'].append(attention_mask)
            seqs_tokenized['word_ids'].append(word_ids)

        seqs_tokenized['input_ids'] = torch.tensor(seqs_tokenized['input_ids'])
        seqs_tokenized['attention_mask'] = torch.tensor(seqs_tokenized['attention_mask'])

        self.seqs_tokenized = seqs_tokenized
        self.word_ids = seqs_tokenized['word_ids']
        self.input_ids = seqs_tokenized['input_ids']
        self.attention_mask = seqs_tokenized['attention_mask']
        assert len(self.input_ids) == len(self.attention_mask)

        labels = examples[1]
        labels_tokenized = []
        labels_tokenized = get_token_labels(labels, self.word_ids, dict_lbl2idx, label_all_tokens)
                
        self.y = torch.tensor(labels_tokenized)
        assert len(self.y) == len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if not self.test:
            item["labels"] = self.y[idx]
        return item
    
    def __len__(self):
        return len(self.input_ids)


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


def get_io_data(csv_path, data_size=None):
    data = pd.read_csv(csv_path, sep='\t').sample(data_size, random_state=2707) if data_size else pd.read_csv(csv_path, sep='\t')
    num_cols = len(list(data.columns))

    seqs = []
    labels = []
    tokenizer = TreebankWordTokenizer()
    for _, row in data.iterrows():
        seq = tokenizer.tokenize(row['text'].replace('“', '"').replace('”', '"').replace('’',"'"))
        seqs.append(seq)

        labels.append(get_binary_labels(seq, row, "span_type_start_index", "span_type_end_index"))
    
    return (seqs, labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1).tolist()
    labels = labels.tolist()

    y_true, y_pred = label_update(labels, preds)
    token_f1, span_f1 = get_metrics(y_true, y_pred)
    return {'token_f1': token_f1, 'span_f1': span_f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--task", type=str, default="NER")
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--model_size", type=int, default=8)
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000, help='maximum number of training epochs')
    args = parser.parse_args()

    if "NER" in args.task:
        dict_lbl2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        dict_idx2lbl = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    elif args.task == "ABSA":
        dict_idx2lbl = {0: 'O', 1: 'B-Aspect', 2: 'I-Aspect', 3: 'B-Opinion', 4: 'I-Opinion'}
        dict_lbl2idx = {'O': 0, 'B-Aspect': 1, 'I-Aspect': 2, 'B-Opinion': 3, 'I-Opinion': 4}
    elif args.task == "DEF":
        dict_lbl2idx = {
            'O': 0, 'B-Alias-Term': 1, 'I-Alias-Term': 2, 'B-Alias-Term-frag': 3, 'I-Alias-Term-frag': 4, 'B-Definition': 5, 
            'I-Definition': 6, 'B-Definition-frag': 7, 'I-Definition-frag': 8, 'B-Ordered-Definition': 9, 'I-Ordered-Definition': 10, 
            'B-Ordered-Term': 11, 'I-Ordered-Term': 12, 'B-Qualifier': 13, 'I-Qualifier': 14, 'B-Referential-Definition': 15, 
            'I-Referential-Definition': 16, 'B-Referential-Term': 17, 'I-Referential-Term': 18, 'B-Secondary-Definition': 19, 
            'I-Secondary-Definition': 20, 'B-Term': 21, 'I-Term': 22, 'B-Term-frag': 23, 'I-Term-frag': 24
        }
        dict_idx2lbl = {
            0: 'O', 1: 'B-Alias-Term', 2: 'I-Alias-Term', 3: 'B-Alias-Term-frag', 4: 'I-Alias-Term-frag', 5: 'B-Definition', 
            6: 'I-Definition', 7: 'B-Definition-frag', 8: 'I-Definition-frag', 9: 'B-Ordered-Definition', 10: 'I-Ordered-Definition', 
            11: 'B-Ordered-Term', 12: 'I-Ordered-Term', 13: 'B-Qualifier', 14: 'I-Qualifier', 15: 'B-Referential-Definition', 
            16: 'I-Referential-Definition', 17: 'B-Referential-Term', 18: 'I-Referential-Term', 19: 'B-Secondary-Definition', 
            20: 'I-Secondary-Definition', 21: 'B-Term', 22: 'I-Term', 23: 'B-Term-frag', 24: 'I-Term-frag'
        }
    else:
        raise ValueError("Invalid task")

    model_names = {
        'mistral': ["mistralai/Mistral-{model_size}B-v0.3", modeling_mistral.UnmaskingMistralForTokenClassification],
        'llama': ["meta-llama/Llama-3.1-{model_size}B", modeling_llama.UnmaskingLlamaForTokenClassification],
        'qwen': ["Qwen/Qwen2.5-{model_size}B", modeling_qwen.UnmaskingQwen2ForTokenClassification],
        'qwen3': ["Qwen/Qwen3-{model_size}B-Base", modeling_qwen3.UnmaskingQwen3ForTokenClassification],
    }

    model_name = model_names[args.model_name][0].format(model_size=args.model_size)
    output_dir = f"models/{args.task}/{args.model_name}-{args.model_size}B-token-classification-unmask"
        
    
    num_labels = len(dict_lbl2idx)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

     # model
    print(output_dir)
    print(f"Loading Model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if args.train:
        model = model_names[args.model_name][1].from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=dict_idx2lbl,
                label2id=dict_lbl2idx,
                ignore_mismatched_sizes=True,
                device_map="auto",
                dtype=torch.bfloat16,
            )
        model.config.pad_token_id = tokenizer.eos_token_id
        lora_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=12,
            bias="none",
            task_type="TOKEN_CLS",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
    else:
        config = PeftConfig.from_pretrained(output_dir)
        base_model = model_names[args.model_name][1].from_pretrained(
            config.base_model_name_or_path,
            num_labels=num_labels,
            id2label=dict_idx2lbl,
            label2id=dict_lbl2idx,
            ignore_mismatched_sizes=True,
            device_map="auto",
            dtype="auto",
        )
        model = PeftModel.from_pretrained(base_model, output_dir)
        model.eval()

    # dataset
    kwargs= {}

    if args.train:
        if args.task == "ABSA":
            train_seqs, train_labels = get_io_data(f'{args.data_path}/{args.task}/train.csv', args.data_size)
            dev_seqs, dev_labels = get_io_data(f'{args.data_path}/{args.task}/dev.csv')
        elif "NER" in args.task or args.task == "DEF":
            if args.data_size:
                train_data = pd.read_csv(f'{args.data_path}/{args.task}/train.csv', sep="\t").sample(args.data_size, random_state=2707)
            else:
                train_data = pd.read_csv(f'{args.data_path}/{args.task}/train.csv', sep="\t")
            if 'tags' not in train_data.columns.tolist():
                train_data['tags'] = train_data['ner_tags'].apply(lambda x: [dict_idx2lbl[t] for t in ast.literal_eval(x)])
                train_seqs, train_labels = train_data['tokens'].apply(lambda x: ast.literal_eval(x)).tolist(), train_data['tags'].tolist() 
            else:
                train_seqs, train_labels = train_data['tokens'].apply(lambda x: ast.literal_eval(x)).tolist(), train_data['tags'].apply(lambda x: ast.literal_eval(x)).tolist() 
            dev_data = pd.read_csv(f'{args.data_path}/{args.task}/dev.csv', sep="\t")
            if 'tags' not in dev_data.columns.tolist():
                dev_data['tags'] = dev_data['ner_tags'].apply(lambda x: [dict_idx2lbl[t] for t in ast.literal_eval(x)])
                dev_seqs, dev_labels = dev_data['tokens'].apply(lambda x: ast.literal_eval(x)).tolist(), dev_data['tags'].tolist()
            else:
                dev_seqs, dev_labels = dev_data['tokens'].apply(lambda x: ast.literal_eval(x)).tolist(), dev_data['tags'].apply(lambda x: ast.literal_eval(x)).tolist()
        trainset = myDataset((train_seqs, train_labels), tokenizer, args.max_seq_len, dict_lbl2idx)
        devset = myDataset((dev_seqs, dev_labels), tokenizer, args.max_seq_len, dict_lbl2idx)

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
            fp16=False,
            bf16=True,
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            metric_for_best_model="token_f1",
            greater_is_better=True,
            save_steps=100,
            logging_steps=100,
            save_total_limit=2,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=devset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        trainer.train()
        trainer.save_model(output_dir)
    else:
        if args.task == "ABSA":
            test_seqs, test_labels = get_io_data(f'{args.data_path}/{args.task}/test.csv')
        elif "NER" in args.task or args.task == "DEF":
            test_data = pd.read_csv(f'{args.data_path}/{args.task}/test.csv', sep="\t")
            if 'tags' not in test_data.columns.tolist():
                test_data['tags'] = test_data['ner_tags'].apply(lambda x: [dict_idx2lbl[t] for t in ast.literal_eval(x)])
                test_seqs, test_labels = test_data['tokens'].apply(lambda x: ast.literal_eval(x)).tolist(), test_data['tags'].tolist() 
            else:
                test_seqs, test_labels = test_data['tokens'].apply(lambda x: ast.literal_eval(x)).tolist(), test_data['tags'].apply(lambda x: ast.literal_eval(x)).tolist() 
        testset = myDataset((test_seqs, test_labels), tokenizer, args.max_seq_len, dict_lbl2idx, test=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=args.bs
        )
        trainer = Trainer(model=model, args=training_args, data_collator=collate_fn)
        predictions = trainer.predict(testset)
        subword_preds = np.argmax(predictions.predictions, axis=-1).tolist()
        subword_labels = testset.y.cpu().detach().tolist() # label_tokenized
        assert len(subword_preds)==len(subword_labels), ipdb.set_trace()

        word_preds, word_labels = get_word_labels(testset, subword_labels, subword_preds)
        word_preds_lb = [[dict_idx2lbl[p] for p in pred] for pred in word_preds]
        word_labels_lb = [[dict_idx2lbl[l] for l in label] for label in word_labels]

        print("**********")
        test_metrics = get_metrics(word_labels, word_preds)
        print(f"Argument Token F1: {test_metrics[0]}, Argument Span F1: {test_metrics[1]}")
        print("SeqEval F1: ",round(get_ner_metrics(word_labels_lb, word_preds_lb)['overall_f1'], 3))