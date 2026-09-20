import os
import torch
import torch.distributed as dist
import argparse
import stanza
import numpy as np
import pandas as pd
from utils import *
from datasets import Dataset, concatenate_datasets, IterableDataset
from trl import SFTTrainer, SFTConfig 
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.data_utils import truncate_dataset, pack_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from typing import Any, Callable, Optional, TypeVar, Union
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from modeling import (
    modeling_llama,
    modeling_mistral,
    modeling_qwen,
    modeling_qwen3,
)
from accelerate import PartialState, logging
from trl.trainer.utils import pad

import warnings
warnings.filterwarnings("ignore")


class LLMTune():
    def __init__(self):
        self.save_path = None
        self.model_name = None
        self.data_folder = "data/"
        self.model_names = {
            'mistral': ["mistralai/Mistral-{model_size}B-v0.3", modeling_mistral.UnmaskingMistralForCausalLM],
            'llama': ["meta-llama/Llama-3.1-{model_size}B", modeling_llama.UnmaskingLlamaForCausalLM],
            'qwen': ["Qwen/Qwen2.5-{model_size}B", modeling_qwen.UnmaskingQwen2ForCausalLM],
            'qwen3': ["Qwen/Qwen3-{model_size}B-Base", modeling_qwen3.UnmaskingQwen3ForCausalLM],
        }
        self.format_functions = {
            'NER': format_function_ner,
            'DEF': format_function_def,
            'ABSA': format_function_absa,
            'SQUAD': format_function_squad,
            'SA': format_function_sa,
        }
        self.tokenizer = None
        self.model = None

    def get_quantization_config(self):
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            bnb_8bit_compute_dtype=torch.bfloat16
        )

    def get_lora_config(self, alpha=32, dropout=0.1, r=12):
        return LoraConfig(
            lora_alpha=alpha,
            lora_dropout=dropout,
            r=r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
    
    def get_sft_config(self, epochs=10, batch_size=2, learning_rate=1e-4):
        return SFTConfig(
            output_dir=self.save_path,
            overwrite_output_dir=True,
            eval_strategy='steps',
            eval_steps=100,
            max_steps=1000,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_strategy="steps",
            save_strategy="steps",
            logging_steps=100,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            bf16=True,
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            max_length=512,
            padding_free=True,
            disable_tqdm=False,
            remove_unused_columns=False, # so that the data collator works properly and returns attention_unmask
        )

    def load_data(self, task_name="NER", data_size=None):
        train_data = pd.read_csv(f'{self.data_folder}/{task_name}/train.csv', sep='\t') if not data_size else pd.read_csv(f'{self.data_folder}/{task_name}/train.csv', sep='\t').sample(data_size, random_state=2705)
        dev_data = pd.read_csv(f'{self.data_folder}/{task_name}/dev.csv', sep='\t')

        train_data = Dataset.from_pandas(train_data)
        dev_data = Dataset.from_pandas(dev_data)

        return train_data, dev_data

    def load_model(self, task_name="NER", model_name='llama', model_size=8, data_size=None):
        self.model_name = self.model_names[model_name][0].format(model_size=model_size)
        self.save_path = f"models/{task_name}/{model_name}-{model_size}B-instruction-unmask" if not data_size else f"models/{task_name}/{model_name}-{model_size}B-instruction-tuned-{data_size}"
        print(self.save_path)

        print(f"LOADING MODEL {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
            dtype="auto",
        )

        self.model = get_peft_model(self.model, self.get_lora_config())
        self.model.enable_input_require_grads()


    def train(self, train_data, dev_data, task_name="NER"):
        self.model.print_trainable_parameters()
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=dev_data,
            peft_config=self.get_lora_config(),
            processing_class=self.tokenizer,
            formatting_func=self.format_functions['NER'] if "NER" in task_name else self.format_functions[task_name],
            args=self.get_sft_config(),
        )

        print(f"Training Model")
        trainer.train()
        print("Saving Model Adapters")
        trainer.save_model(self.save_path)

    def merge_and_save_model(self):
        print("Merging Adapters with Base Model")
        peft_model = self.model.merge_and_unload()
        peft_model.save_pretrained(self.save_path, safe_serialization=True)
        self.tokenizer.save_pretrained(self.save_path)
        print("Saved merged model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="NER")
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--model_size", type=int, default=8)
    parser.add_argument("--data_size", type=int, default=None)
    args = parser.parse_args()

    tune = LLMTune()
    train_data, dev_data = tune.load_data(args.task, args.data_size)  #read data
    tune.load_model(args.task, args.model_name, args.model_size, args.data_size)  #load model
    tune.train(train_data, dev_data, args.task)  #train model
    # tune.merge_and_save_model()