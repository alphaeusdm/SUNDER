import os
import torch
import torch.nn as nn
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
from unmask_tokenizer import UnmaskTokenizer
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


START_TOKEN = "<start>"
END_TOKEN = "<end>"

START_TOKEN_ID = None
END_TOKEN_ID = None

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        if 'position_ids' in output:
            del output['position_ids']
        if "attention_mask" not in output:
            input_ids = [torch.tensor(example["input_ids"]) for example in examples]
            attention_mask = [torch.ones_like(ids) for ids in input_ids]
            if self.padding_free:
                attention_mask = [torch.cat(attention_mask, dim=0)]
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        if "attention_unmask" in examples[0]:
            attention_unmask = [torch.tensor(example["attention_unmask"]) for example in examples]
            if self.padding_free:
                attention_unmask = [torch.cat(attention_unmask, dim=0)]

            output["attention_unmask"] = pad(
                attention_unmask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )

        return output


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, original_vocab_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_vocab_size = original_vocab_size
        self.hooks_registered = False
        if original_vocab_size is not None:
            self.original_vocab_indices = torch.arange(original_vocab_size)


    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        packing = True
        dataset_initial = super()._prepare_dataset(dataset, processing_class, args, packing, formatting_func, dataset_name)
        
        def get_attention_unmask(example):
            input_ids = example['input_ids']
            is_tensor = isinstance(input_ids, torch.Tensor)
            device = input_ids.device if is_tensor else None

            if is_tensor:
                input_ids = input_ids.cpu().numpy()
            else:
                input_ids = np.array(input_ids)

            start_positions = np.where(input_ids.flatten()==START_TOKEN_ID)[0]
            end_positions = np.where(input_ids.flatten()==END_TOKEN_ID)[0]

            if len(start_positions) > len(end_positions):
                end_positions = np.append(end_positions, len(input_ids)-1)

            attention_unmask = np.zeros(input_ids.shape, dtype=int)
            for start, end in zip(start_positions, end_positions):
                attention_unmask[...,start:end+1] = 1

            if is_tensor:
                attention_unmask = torch.tensor(attention_unmask).to(device)
            else:
                attention_unmask = attention_unmask.tolist()

            example['attention_unmask'] = attention_unmask

            return example

        dataset_initial = dataset_initial.map(
            get_attention_unmask,
        )

        return dataset_initial
    
    def _setup_embedding_hooks(self):
        input_embeds  = self.model.get_input_embeddings()
        output_embeds = self.model.get_output_embeddings()
        
        input_embeds.weight.requires_grad  = True
        output_embeds.weight.requires_grad = True
        
        verified = [False]  # use list for mutability inside lambda
            
        def input_hook(grad):
            # Zero out first
            grad = grad.index_fill_(
                0, self.original_vocab_indices.to(grad.device), 0)
            
            # Then verify
            if not verified[0]:
                orig_max = grad[:self.original_vocab_size].abs().max().item()
                new_max  = grad[self.original_vocab_size:].abs().max().item()
                print(f"=== Gradient Check (after zeroing) ===")
                print(f"Original rows grad max: {orig_max:.8f}  (should be 0.0)")
                print(f"New token rows grad max: {new_max:.8f}  (should be > 0.0)")
                if orig_max == 0.0 and new_max > 0.0:
                    print("✓ Hooks working correctly")
                verified[0] = True
            
            return grad
        
        input_embeds.weight.register_hook(input_hook)
        output_embeds.weight.register_hook(
            lambda grad: grad.index_fill_(
                0, self.original_vocab_indices.to(grad.device), 0)
        )
        print(f"Embedding hooks registered — only rows {self.original_vocab_size}+ will be updated")

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.hooks_registered and self.original_vocab_size is not None:
            self._setup_embedding_hooks()
            self.hooks_registered = True
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        if self.state.global_step == 1:
            grad = self.model.get_input_embeddings().weight.grad
            if grad is not None:
                orig_max = grad[:self.original_vocab_size].abs().max().item()
                new_max  = grad[self.original_vocab_size:].abs().max().item()
                print(f"Original rows grad max: {orig_max:.8f}  (should be 0.0)")
                print(f"New token rows grad max: {new_max:.8f}  (should be > 0.0)")
            else:
                print("WARNING: grad is None after first step")
        
        return loss


class LLMTune():
    def __init__(self):
        self.save_path = None
        self.model_name = None
        self.data_folder = "data/"
        self.model_names = {
            'mistral': ["mistralai/Mistral-{model_size}B-v0.3", modeling_mistral.SelectiveUnmaskingMistralForCausalLM],
            'llama': ["meta-llama/Llama-3.1-{model_size}B", modeling_llama.SelectiveUnmaskingLlamaForCausalLM],
            'qwen': ["Qwen/Qwen2.5-{model_size}B", modeling_qwen.SelectiveUnmaskingQwen2ForCausalLM],
            'qwen3': ["Qwen/Qwen3-{model_size}B-Base", modeling_qwen3.SelectiveUnmaskingQwen3ForCausalLM],
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
        self.original_vocab_size = None

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
                        # "gate_proj", "up_proj", "down_proj"],
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
            disable_tqdm=False,
            remove_unused_columns=False, # so that the data collator works properly and returns attention_unmask
        )

    def load_data(self, task_name="NER", data_size=None):
        train_data = pd.read_csv(f'{self.data_folder}/{task_name}/train.csv', sep='\t') if not data_size else pd.read_csv(f'{self.data_folder}/{task_name}/train.csv', sep='\t').sample(data_size, random_state=2705)
        dev_data = pd.read_csv(f'{self.data_folder}/{task_name}/dev.csv', sep='\t')

        # if "NER" in task_name:
        #     if 'lang' in train_data.columns:
        #         stanza.download(train_data.lang.unique()[0])
                
        train_data = Dataset.from_pandas(train_data)
        dev_data = Dataset.from_pandas(dev_data)

        return train_data, dev_data

    def unfreeze_new_embeddings(self, model, original_vocab_size):
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True
                
                def make_hook(orig_size):
                    print(orig_size)
                    def hook(grad):
                        grad_clone = grad.clone()
                        grad_clone[:orig_size] = 0
                        return grad_clone
                    return hook
                
                param.register_hook(make_hook(original_vocab_size))


    def load_model(self, task_name="NER", model_name='llama', model_size=8, data_size=None):
        self.model_name = self.model_names[model_name][0].format(model_size=model_size)
        self.save_path = f"models/{task_name}/{model_name}-{model_size}B-instruction-bimask" if not data_size else f"models/{task_name}/{model_name}-{model_size}B-instruction-bimask-{data_size}"
        print(self.save_path)
        print(f"LOADING MODEL {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.original_vocab_size = len(self.tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = self.model_names[model_name][1].from_pretrained(
            self.model_name,
            # quantization_config=self.get_quantization_config(),
            # use_cache=False,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
            dtype="auto",
        )

        special_tokens = {"additional_special_tokens": [START_TOKEN, END_TOKEN]}
        self.tokenizer.add_special_tokens(special_tokens)    
        self.model.resize_token_embeddings(len(self.tokenizer))

        global START_TOKEN_ID
        global END_TOKEN_ID
        START_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(START_TOKEN)
        END_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(END_TOKEN)

        with torch.no_grad():
            input_embeddings = self.model.get_input_embeddings().weight
            output_embeddings = self.model.get_output_embeddings().weight

            if "qwen" in model_name:
                bos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            else:
                bos_token_id = self.tokenizer.bos_token_id
                eos_token_id = self.tokenizer.eos_token_id

            input_embeddings[START_TOKEN_ID] = input_embeddings[bos_token_id].detach().clone()
            input_embeddings[END_TOKEN_ID] = input_embeddings[eos_token_id].detach().clone()
            output_embeddings[START_TOKEN_ID] = output_embeddings[bos_token_id].detach().clone()
            output_embeddings[END_TOKEN_ID] = output_embeddings[eos_token_id].detach().clone()

        # self.model = prepare_model_for_kbit_training(self.model)
        # self.unfreeze_new_embeddings(self.model, original_vocab_size)

        self.model = get_peft_model(self.model, self.get_lora_config())

        def diagnose_trainable(model, threshold_m=1):
            print("\n=== All Trainable Parameters ===")
            total_trainable = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    n = param.numel()
                    total_trainable += n
                    if n > threshold_m * 1e6:
                        print(f"{name}: {param.shape} = {n/1e6:.1f}M")
            print(f"\nTotal trainable: {total_trainable/1e9:.2f}B")

        diagnose_trainable(self.model, threshold_m=1)


    def save_new_embeddings(self, model, save_path, original_vocab_size):
        input_embeds  = model.get_input_embeddings().weight[original_vocab_size:].detach().cpu()
        output_embeds = model.get_output_embeddings().weight[original_vocab_size:].detach().cpu()
        
        torch.save(
            {
                "new_input_embeddings":  input_embeds,
                "new_output_embeddings": output_embeds,
                "original_vocab_size":   original_vocab_size,
            },
            os.path.join(save_path, "new_embeddings.pt")
        )
        print(f"Saved new token embeddings to {save_path}/new_embeddings.pt")

    def train(self, train_data, dev_data, task_name="NER"):
        self.model.print_trainable_parameters()
        trainer = CustomSFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=dev_data,
            data_collator = CustomDataCollatorForLanguageModeling(
                pad_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
                completion_only_loss=self.get_sft_config().completion_only_loss,
                padding_free=True,
                pad_to_multiple_of=self.get_sft_config().pad_to_multiple_of,
            ),
            peft_config=self.get_lora_config(),
            processing_class=self.tokenizer,
            formatting_func=self.format_functions['NER'] if "NER" in task_name else self.format_functions[task_name],
            args=self.get_sft_config(),
            original_vocab_size=self.original_vocab_size,
            # callbacks=[VerifyHooksCallback(self.model, self.original_vocab_size)],
        )

        print(f"Training Model")
        trainer.train()
        print("Saving Model Adapters")
        trainer.save_model(self.save_path)
        self.save_new_embeddings(self.model, self.save_path, self.original_vocab_size)
        # self.tokenizer.save_pretrained(self.save_path)


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