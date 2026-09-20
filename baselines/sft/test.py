import torch
import re
import os # remove after testing
import json
import torch
import faiss_index
import argparse
import stanza
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm
from metrics import *
from sklearn.metrics import f1_score
from peft import PeftConfig, PeftModel
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import warnings
warnings.filterwarnings("ignore")


def load_model(task, model_name, model_size, tuned=False, data_size=None, instruction_tune=False):
    model_names = {
        'mistral': f"mistralai/Mistral-{model_size}B-v0.3",
        'llama': f"meta-llama/Llama-3.1-{model_size}B",
        'qwen': f"Qwen/Qwen2.5-{model_size}B",
        'qwen3': f"Qwen/Qwen3-{model_size}B-Base",
        "llama-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen3-instruct": "Qwen/Qwen3-8B",
        "tuned": f"./models/{task}/{model_name}-{model_size}B-instruction-tuned" if not data_size else f"./models/{task}/{model_name}-{model_size}B-instruction-tuned-{data_size}"
    }
    if tuned:
        model = model_names["tuned"]
    else:
        model = model_names[model_name]

    print(f"Loading {model}")
    if tuned:
        tokenizer = AutoTokenizer.from_pretrained(model)
        config = PeftConfig.from_pretrained(model)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager"
            )
        model = PeftModel.from_pretrained(base_model,model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager"
            )

    return model, tokenizer


def load_prompt(name):
    hi_flag = False
    if "NER" in name:
        if "HI" in name: hi_flag = True
        name="NER"
    with open(f"prompts/{name.lower()}.json", 'r') as fp:
        prompt = json.load(fp)
    
    return prompt['a_prompt'] if hi_flag else prompt['prompt']


def get_few_shot_examples(task_name, indices):
    data = pd.read_csv(f'data/{task_name}/train.csv', sep='\t').iloc[indices]
    columns = list(data.keys())[1:]
    if "NER" in task_name: task_name="NER"
    tasks = {
        "NER": get_ner_example,
        "DEF": get_def_example,
        "ABSA": get_absa_example,
        "SQUAD": get_squad_example,
        "SA": get_sa_example,
    }
    examples = ""
    if task_name == "SQUAD":
        for ind, row in data.iterrows():
            examples += f"""
Input {ind+1}:
Question: {row['question']}
Context: {row['context']}

Response {ind+1}:
{tasks[task_name](row)}"""
    elif task_name == "SA":
        for ind, row in data.iterrows():
            examples += f"""
Input {ind+1}:
Text: {row['text']}

Response {ind+1}:
{tasks[task_name](row)}"""
    else:
        for ind, row in data.iterrows():
            examples += f"""
Input {ind+1}:
Text: {row['text']}

Response {ind+1}:
{tasks[task_name](row, columns)}"""

    return examples


def load_test_data(task_name):
    return pd.read_csv(f'data/{task_name}/test.csv', sep='\t').sample(10, random_state=1234)


def test(task_name, test_data, model, tokenizer, model_size, model_name, tuned=False, cot=False, num_examples=None, data_size=None):
    examples = None
    if task_name != "SQUAD":
        columns = list(test_data.columns)[1:]
    gold_outputs = list()
    predicted_outputs = list()
    outputs = list()

    instruction_prompt = load_prompt(task_name)

    if num_examples:
        print(num_examples)
        if task_name != "SQUAD":
            template = "{instruction}\n\nExamples:{examples}\n\nInput:\nText: {sentence}\n\nResponse:\n"
            prompt = PromptTemplate(template=template, input_variables=["instruction","examples","sentence"])
        else:
            template = "{instruction}\n\nExamples:{examples}\n\nInput:\nQuestion: {question}\nContext: {context}\n\nResponse:\n"
            prompt = PromptTemplate(template=template, input_variables=["instruction","examples","question","context"])
        print("Creating Index")
        example_index = faiss_index.FaissIndex(task_name=task_name)
        example_index.setup()
    else:
        if task_name != "SQUAD":
            template = "{instruction}\n\nInput:\nText: {sentence}\n\nResponse:\n"
            prompt = PromptTemplate(template=template, input_variables=["instruction","sentence"])
        else:
            template = "{instruction}\n\nInput:\nQuestion: {question}\nContext: {context}\n\nResponse:\n"
            prompt = PromptTemplate(template=template, input_variables=["instruction","question","context"])

    for ind, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        if task_name not in ["SQUAD", "SA"]:
            if "NER" in task_name:
                bin_argument = get_gold_labels("NER", row, columns)
            else:
                bin_argument = get_gold_labels(task_name, row, columns)
            gold_outputs.append(bin_argument)
            sentence = row['text']
        elif task_name == "SA":
            gold_outputs.append(row["label_text"])
            sentence = row['text']
        else:
            gold_outputs.append(row["answer_text"])
            question = row['question']
            context = row['context']
        if num_examples:
            if task_name == "SQUAD":
                examples = get_few_shot_examples(task_name, example_index.search_relevant(question,num_examples))
                generate_prompt = prompt.format(instruction=instruction_prompt, examples=examples, question=question, context=context)
            else:
                examples = get_few_shot_examples(task_name, example_index.search_relevant(sentence,num_examples))
                generate_prompt = prompt.format(instruction=instruction_prompt, examples=examples, sentence=sentence)
        else:
            if task_name == "SQUAD":
                generate_prompt = prompt.format(instruction=instruction_prompt, question=question, context=context)
            else:
                generate_prompt = prompt.format(instruction=instruction_prompt, sentence=sentence)
            
        model_input = tokenizer(generate_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**model_input,
                                    max_new_tokens=500,
                                    temperature=1e-5,
                                    do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    )
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            print(output)
        output = output[re.search(r"Response:\n",output).end():].strip()
        output = re.sub(r"\n\n",r"\n",output)
        if task_name == "SQUAD":
            outputs.append({'question': question, 'context': context, 'model output': output})
        else:
            outputs.append({'text': sentence, 'model output': output})
        output = output.strip().split('\n')

        if task_name not in ["SQUAD", "SA"]:
            if "NER" in task_name:
                bin_argument = get_spans("NER", row, output)
            else:
                bin_argument = get_spans(task_name, row, output)
            predicted_outputs.append(bin_argument)
        else:
            predicted_outputs.append(get_answers(output))

        if task_name not in ["SQUAD", "SA"]:
            if len(gold_outputs[-1]) != len(predicted_outputs[-1]):
                print(ind)
                gold_outputs = gold_outputs[:-1]
                predicted_outputs = predicted_outputs[:-1]

    file_name = str(num_examples) if num_examples else "it" if tuned else f"progress-{data_size}" if data_size else 'zero'
    output_folder_path = f"outputs/{task_name}/{model_name}"
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)
    with open(f'{output_folder_path}/{file_name}-{model_size}-test-tuned.json', 'w') as f:
        json.dump(outputs, f)

    return gold_outputs, predicted_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="NER")
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--model_size", type=int, default=8)
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--instruction_tune", type=bool, default=False)
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()

    if args.task == "CSI" or args.task == "TSD":
        dict_idx2lbl = {2:'I', 1:'B', 0:'O'}
    elif args.task == "ABSA":
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
    
    model, tokenizer = load_model(args.task, args.model_name, args.model_size, args.tuned, args.data_size, args.instruction_tune)
    test_data = load_test_data(args.task)
    gold_outputs, predicted_outputs = test(args.task, test_data, model, tokenizer, args.model_size, args.model_name, args.tuned, args.cot, args.num_examples, args.data_size)
    if args.task not in ["SQUAD", "SA"]:
        predicted_outputs_lb = [[dict_idx2lbl[p] for p in pred] for pred in predicted_outputs]
        gold_outputs_lb = [[dict_idx2lbl[l] for l in label] for label in gold_outputs]
        output_scores = get_span_metrics(gold_outputs, predicted_outputs, dict_idx2lbl)
        print("Token F1: {}, Span F1: {}".format(output_scores[0], output_scores[1]))
        metrics = get_seqeval_metrics(gold_outputs_lb, predicted_outputs_lb)
        for k, v in metrics.items():
            if isinstance(v,dict):
                print(f"{k}: {round(v['f1'],3)}")
    elif args.task == "SQUAD":
        output_scores = get_mhqa_metrics(gold_outputs, predicted_outputs)
        print(f"EM: {output_scores['em']}, F1: {output_scores['f1']}")
    else:
        output_scores = get_f1_score(gold_outputs, predicted_outputs)
        print(f"F1: {output_scores}")
