import re
import json
import pandas as pd
import ast
import re
import html
from sacremoses import MosesTokenizer, MosesDetokenizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import stanza
from typing import List, Optional

PUNC_MATCH = re.compile(r'[.,-/|]')

class MultilingualProcessor:    
    def __init__(self, lang: Optional[str] = None, detokenize=False):
        self.lang = lang
        self.nlp = MosesDetokenizer(lang=lang) if detokenize else MosesTokenizer(lang=lang)
        self.protected = [r'\d+:\d+',r"'s",r"\w+'\s"] if lang in ['fr', 'it'] else [r'\d+:\d+',r"'s",r"\w+'\w+"]

    def split_punctuation(self, word):
        trailing = []
        while word:
            if PUNC_MATCH.match(word[-1]):
                trailing.append(word[-1])
            else:
                break
            word = word[:-1]
                
        return word, trailing

    def tokenize(self, text: str) -> List[str]:
        if self.lang not in ['si', 'hi']:
            tokenized = self.nlp.tokenize(text, protected_patterns=self.protected)
            tokenized = [html.unescape(token) for token in tokenized]

            if len(tokenized[-1]) > 1 and '.' in tokenized[-1]:
                token = tokenized.pop(-1)
                tokenized.append(token[:-1])
                tokenized.append(token[-1])
        else:
            tokenized = []
            for word in text.split():
                word, trailing = self.split_punctuation(word)
                if word:
                    tokenized.append(word)
                if trailing:
                    for char in reversed(trailing):
                        tokenized.append(char)
        
        return tokenized
    
    def detokenize(self, tokens: List[str]) -> str:
        if self.lang not in ['si', 'hi']:
            tokens = [token.replace('“', '"').replace('”', '"').replace('’',"'").replace("``",'"').replace("''", '"') for token in tokens]
            text = self.nlp.detokenize(tokens)
        else:
            text = ""
            for i, word in enumerate(tokens):
                if i == 0:
                    text += word
                elif PUNC_MATCH.match(word)and text[-1] != 'ු':
                    text += word
                else:
                    text += " " + word
            text = text.strip()
        return text



def load_prompt(name, lang='en'):
    with open(f"prompts/{name.lower()}.json", 'r') as fp:
        prompt = json.load(fp)
    return prompt['a_prompt'] if lang == 'hi' else prompt['prompt']


def format_function_squad(sample):
    prompt = load_prompt('squad')
    input = f"{prompt}\n\nInput:\nQuestion: {sample['question']}\nContext: {sample['context']}\n\nResponse:\nAnswer: {sample['answer_text']}\n"
    return input


def format_function_sa(sample):
    prompt = load_prompt('sa')
    input = f"{prompt}\n\nInput:\nText: {sample['text']}\n\nResponse:\nSentiment Label: {sample['label_text']}\n"
    return input


def format_function_absa(sample):
    columns = list(sample.keys())[1:]
    prompt = load_prompt('absa')
    input = f"{prompt}\n\nInput:\nText: {sample['text']}\n\nResponse:\n"
    for column in columns:
        if 'Opinion' in column and sample[column]:
            input+=column+": "+sample[column]+"\n"
        if 'Aspect' in column and sample[column]:
            input+=column+": "+sample[column]+"\n"
    return input


def format_function_ner(sample):
    prompt = load_prompt('ner', lang='hi') if 'lang' in sample and sample['lang'] == 'hi' else load_prompt('ner')
    input = f"{prompt}\n\nInput:\nText: {sample['text']}\n\nResponse:\n"

    if 'lang' in sample and sample['lang'] != 'en':
        # tokenizer = MosesDetokenizer(lang=sample['lang'])
        tokenizer = MultilingualProcessor(sample['lang'], detokenize=True)
    else:
        tokenizer = TreebankWordDetokenizer()
    tokens = ast.literal_eval(sample['tokens'])
    id_label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    tag_count = {'PER': 0, 'LOC': 0, 'ORG': 0, 'MISC': 0}
    tag_to_string = {'PER': 'Person', 'LOC': 'Location', 'ORG': 'Organization', 'MISC': 'Miscellaneous Name'}
    start_id = None
    tags = ast.literal_eval(sample['ner_tags'])
    for i, tag in enumerate(tags):
        if tag == 0:
            if start_id != None:
                span = tokenizer.tokenize(tokens[start_id:i]) if isinstance(tokenizer,TreebankWordDetokenizer) else tokenizer.detokenize(tokens[start_id:i])
                tag_name = id_label_map[tags[start_id]].split('-')[1]
                input+=f"{tag_to_string[tag_name]} {tag_count[tag_name]+1}: {span}\n"
                tag_count[tag_name] += 1
                start_id = None
            else:
                continue
        else:
            if start_id == None:
                start_id = i
            else:
                if tags[start_id] == tag-1:
                    continue
                else:
                    span = tokenizer.tokenize(tokens[start_id:i]) if isinstance(tokenizer,TreebankWordDetokenizer) else tokenizer.detokenize(tokens[start_id:i])
                    tag_name = id_label_map[tags[start_id]].split('-')[1]
                    input+=f"{tag_to_string[tag_name]} {tag_count[tag_name]+1}: {span}\n"
                    tag_count[tag_name] += 1
                    start_id = i
    if start_id != None:
        span = tokenizer.tokenize(tokens[start_id:]) if isinstance(tokenizer,TreebankWordDetokenizer) else tokenizer.detokenize(tokens[start_id:])
        tag_name = id_label_map[tags[start_id]].split('-')[1]
        input+=f"{tag_to_string[tag_name]} {tag_count[tag_name]+1}: {span}\n"
        tag_count[tag_name] += 1

    return input


def format_function_def(sample):
    prompt = load_prompt('def')
    input = f"{prompt}\n\nInput:\nText: {sample['text']}\n\nResponse:\n"

    tokenizer = TreebankWordDetokenizer()
    tokens = ast.literal_eval(sample['tokens'])
    id_label_map = {
        0: 'O', 1: 'B-Alias-Term', 2: 'I-Alias-Term', 3: 'B-Alias-Term-frag', 4: 'I-Alias-Term-frag', 5: 'B-Definition', 
        6: 'I-Definition', 7: 'B-Definition-frag', 8: 'I-Definition-frag', 9: 'B-Ordered-Definition', 10: 'I-Ordered-Definition', 
        11: 'B-Ordered-Term', 12: 'I-Ordered-Term', 13: 'B-Qualifier', 14: 'I-Qualifier', 15: 'B-Referential-Definition', 
        16: 'I-Referential-Definition', 17: 'B-Referential-Term', 18: 'I-Referential-Term', 19: 'B-Secondary-Definition', 
        20: 'I-Secondary-Definition', 21: 'B-Term', 22: 'I-Term', 23: 'B-Term-frag', 24: 'I-Term-frag'
    }
    tag_count = {
        'Qualifier': 0, 'Definition': 0, 'Ordered Definition': 0, 'Referential Term': 0, 'Term frag': 0, 
        'Alias Term frag': 0, 'Definition frag': 0, 'Alias Term': 0, 'Secondary Definition': 0, 'Term': 0, 
        'Referential Definition': 0, 'Ordered Term': 0
    }
    start_id = None
    tags = ast.literal_eval(sample['def_tags'])
    for i, tag in enumerate(tags):
        if tag == 0:
            if start_id != None:
                span = tokenizer.tokenize(tokens[start_id:i])
                tag_name = id_label_map[tags[start_id]].split('-')
                tag_name = " ".join(tag_name[1:])
                input+=f"{tag_name} {tag_count[tag_name]+1}: {span}\n"
                tag_count[tag_name] += 1
                start_id = None
            else:
                continue
        else:
            if start_id == None:
                start_id = i
            else:
                if tags[start_id] == tag-1:
                    continue
                else:
                    span = tokenizer.tokenize(tokens[start_id:i])
                    tag_name = id_label_map[tags[start_id]].split('-')
                    tag_name = " ".join(tag_name[1:])
                    input+=f"{tag_name} {tag_count[tag_name]+1}: {span}\n"
                    tag_count[tag_name] += 1
                    start_id = i
    if start_id != None:
        span = tokenizer.tokenize(tokens[start_id:])
        tag_name = id_label_map[tags[start_id]].split('-')
        tag_name = " ".join(tag_name[1:])
        input+=f"{tag_name} {tag_count[tag_name]+1}: {span}\n"
        tag_count[tag_name] += 1

    return input


def get_binary_list_ner(tokenized_text, tags, spans, span_type):
    label_id_map = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    for span in spans:
        span_len = len(span)
        for i in range(len(tokenized_text)-span_len+1):
            if tokenized_text[i:i+span_len]==span and tags[i:i+span_len]==[0]*span_len:
                tags[i:i+span_len] = [label_id_map[f'B-{span_type}']]+[label_id_map[f'I-{span_type}']]*(span_len-1)
    return tags


def get_entities_ner(row, predictions):
    text = row['text']
    if 'lang' in row and row['lang'] != 'en':
        # tokenizer = MosesTokenizer(lang=row['lang'])
        tokenizer = MultilingualProcessor(lang=row['lang'])
    else:
        tokenizer = TreebankWordTokenizer()
    id_label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    spans = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
    for span in predictions:
        span = span.split(': ')
        if len(span)>1:
            if 'person' in span[0].lower() and span[1].strip():
                spans['PER'].append(tokenizer.tokenize(span[1].replace('“', '"').replace('”', '"').replace('’',"'")))
            elif 'location' in span[0].lower() and span[1].strip():
                spans['LOC'].append(tokenizer.tokenize(span[1].replace('“', '"').replace('”', '"').replace('’',"'")))
            elif 'organization' in span[0].lower() and span[1].strip():
                spans['ORG'].append(tokenizer.tokenize(span[1].replace('“', '"').replace('”', '"').replace('’',"'")))
            elif 'miscellaneous name' in span[0].lower() and span[1].strip():
                spans['MISC'].append(tokenizer.tokenize(span[1].replace('“', '"').replace('”', '"').replace('’',"'")))
    # tokenized_text = tokenizer.tokenize(text)
    tokenized_text = tokenizer.tokenize(text.replace('“', '"').replace('”', '"').replace('’',"'"))
    tags = [0]*len(tokenized_text)
    for span_type in spans.keys():
        if len(span_type) != 0:
            tags = get_binary_list_ner(tokenized_text, tags, spans[span_type], span_type)

    # tags = [id_label_map[tag] for tag in tags]
    return tags


def get_gold_labels_ner(row, columns):
    id_label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    tags = ast.literal_eval(row['ner_tags'])
    # tags = [id_label_map[tag] for tag in tags]
    return tags


def get_binary_list_def(tokenized_text, tags, spans, span_type):
    label_id_map = {
        'O': 0, 'B-Alias-Term': 1, 'I-Alias-Term': 2, 'B-Alias-Term-frag': 3, 'I-Alias-Term-frag': 4, 'B-Definition': 5, 
        'I-Definition': 6, 'B-Definition-frag': 7, 'I-Definition-frag': 8, 'B-Ordered-Definition': 9, 'I-Ordered-Definition': 10, 
        'B-Ordered-Term': 11, 'I-Ordered-Term': 12, 'B-Qualifier': 13, 'I-Qualifier': 14, 'B-Referential-Definition': 15, 
        'I-Referential-Definition': 16, 'B-Referential-Term': 17, 'I-Referential-Term': 18, 'B-Secondary-Definition': 19, 
        'I-Secondary-Definition': 20, 'B-Term': 21, 'I-Term': 22, 'B-Term-frag': 23, 'I-Term-frag': 24
    }
    for span in spans:
        span_len = len(span)
        for i in range(len(tokenized_text)-span_len+1):
            if tokenized_text[i:i+span_len]==span and tags[i:i+span_len]==[0]*span_len:
                tags[i:i+span_len] = [label_id_map[f'B-{span_type}']]+[label_id_map[f'I-{span_type}']]*(span_len-1)
    return tags


def get_entities_def(row, predictions):
    text = row['text']
    tokenizer = TreebankWordTokenizer()
    id_label_map = {
        0: 'O', 1: 'B-Alias-Term', 2: 'I-Alias-Term', 3: 'B-Alias-Term-frag', 4: 'I-Alias-Term-frag', 5: 'B-Definition', 
        6: 'I-Definition', 7: 'B-Definition-frag', 8: 'I-Definition-frag', 9: 'B-Ordered-Definition', 10: 'I-Ordered-Definition', 
        11: 'B-Ordered-Term', 12: 'I-Ordered-Term', 13: 'B-Qualifier', 14: 'I-Qualifier', 15: 'B-Referential-Definition', 
        16: 'I-Referential-Definition', 17: 'B-Referential-Term', 18: 'I-Referential-Term', 19: 'B-Secondary-Definition', 
        20: 'I-Secondary-Definition', 21: 'B-Term', 22: 'I-Term', 23: 'B-Term-frag', 24: 'I-Term-frag'
    }
    spans = {
        'Qualifier': [], 'Definition': [], 'Ordered Definition': [], 'Referential Term': [], 'Term frag': [], 
        'Alias Term frag': [], 'Definition frag': [], 'Alias Term': [], 'Secondary Definition': [], 'Term': [], 
        'Referential Definition': [], 'Ordered Term': []
    }
    for span in predictions:
        span = span.split(': ')
        if len(span)>1:
            span_tokens = tokenizer.tokenize(span[1].strip().replace('“', '"').replace('”', '"').replace('’',"'"))
            if 'qualifier' in span[0].lower() and span[1].strip():
                spans['Qualifier'].append(span_tokens)
            elif 'definition' in span[0].lower() and span[1].strip():
                spans['Definition'].append(span_tokens)
            elif 'ordered definition' in span[0].lower() and span[1].strip():
                spans['Ordered Definition'].append(span_tokens)
            elif 'referential term' in span[0].lower() and span[1].strip():
                spans['Referential Term'].append(span_tokens)
            elif 'term frag' in span[0].lower() and span[1].strip():
                spans['Term frag'].append(span_tokens)
            elif 'alias term frag' in span[0].lower() and span[1].strip():
                spans['Alias Term frag'].append(span_tokens)
            elif 'definition frag' in span[0].lower() and span[1].strip():
                spans['Definition frag'].append(span_tokens)
            elif 'alias term' in span[0].lower() and span[1].strip():
                spans['Alias Term'].append(span_tokens)
            elif 'secondary definition' in span[0].lower() and span[1].strip():
                spans['Secondary Definition'].append(span_tokens)
            elif 'term' in span[0].lower() and span[1].strip():
                spans['Term'].append(span_tokens)
            elif 'referential definition' in span[0].lower() and span[1].strip():
                spans['Referential Definition'].append(span_tokens)
            elif 'ordered term' in span[0].lower() and span[1].strip():
                spans['Ordered Term'].append(span_tokens)
    # tokenized_text = tokenizer.tokenize(text)
    tokenized_text = tokenizer.tokenize(text.replace('“', '"').replace('”', '"').replace('’',"'"))
    tags = [0]*len(tokenized_text)
    for span_type in spans.keys():
        if len(span_type) != 0:
            tags = get_binary_list_def(tokenized_text, tags, spans[span_type], "-".join(span_type.split()))

    # tags = [id_label_map[tag] for tag in tags]
    return tags


def get_gold_labels_def(row, columns):
    id_label_map = {
        0: 'O', 1: 'B-Alias-Term', 2: 'I-Alias-Term', 3: 'B-Alias-Term-frag', 4: 'I-Alias-Term-frag', 5: 'B-Definition', 
        6: 'I-Definition', 7: 'B-Definition-frag', 8: 'I-Definition-frag', 9: 'B-Ordered-Definition', 10: 'I-Ordered-Definition', 
        11: 'B-Ordered-Term', 12: 'I-Ordered-Term', 13: 'B-Qualifier', 14: 'I-Qualifier', 15: 'B-Referential-Definition', 
        16: 'I-Referential-Definition', 17: 'B-Referential-Term', 18: 'I-Referential-Term', 19: 'B-Secondary-Definition', 
        20: 'I-Secondary-Definition', 21: 'B-Term', 22: 'I-Term', 23: 'B-Term-frag', 24: 'I-Term-frag'
    }
    tags = ast.literal_eval(row['def_tags'])
    # tags = [id_label_map[tag] for tag in tags]
    return tags


def get_binary_list_absa(tokenized_text, tags, spans, span_type):
    label_id_map = {'O': 0, 'B-Aspect': 1, 'I-Aspect': 2, 'B-Opinion': 3, 'I-Opinion': 4}
    for span in spans:
        span_len = len(span)
        for i in range(len(tokenized_text)-span_len+1):
            if tokenized_text[i:i+span_len]==span and tags[i:i+span_len]==[0]*span_len:
                tags[i:i+span_len] = [label_id_map[f'B-{span_type}']]+[label_id_map[f'I-{span_type}']]*(span_len-1)
    return tags

def get_aspect_opinions_absa(row, predictions):
    text = row['text']
    tokenizer = TreebankWordTokenizer()
    id_label_map = {0: 'O', 1: 'B-Aspect', 2: 'I-Aspect', 3: 'B-Opinion', 4: 'I-Opinion'}
    spans = {'Aspect': [], 'Opinion': []}
    for span in predictions:
        span = span.split(': ')
        if len(span)>1:
            if 'aspect' in span[0].lower() and span[1].strip():
                spans['Aspect'].append(tokenizer.tokenize(span[1].replace('“', '"').replace('”', '"').replace('’',"'")))
            elif 'opinion' in span[0].lower() and span[1].strip():
                spans['Opinion'].append(tokenizer.tokenize(span[1].replace('“', '"').replace('”', '"').replace('’',"'")))
    tokenized_text = tokenizer.tokenize(text.replace('“', '"').replace('”', '"').replace('’',"'"))
    tags = [0]*len(tokenized_text)
    for span_type in spans.keys():
        if len(span_type) != 0:
            tags = get_binary_list_absa(tokenized_text, tags, spans[span_type], span_type)

    return tags


def get_gold_labels_absa(row, columns):
    text = row['text']

    tokenizer = TreebankWordTokenizer()
    id_label_map = {0: 'O', 1: 'B-Aspect', 2: 'I-Aspect', 3: 'B-Opinion', 4: 'I-Opinion'}
    spans = {'Aspect': [], 'Opinion': []}
    for column in columns:
        if 'aspect' in column.lower() and not pd.isna(row[column]):
            spans['Aspect'].append(tokenizer.tokenize(row[column].replace('“', '"').replace('”', '"').replace('’',"'")))
        elif 'opinion' in column.lower() and not pd.isna(row[column]):
            spans['Opinion'].append(tokenizer.tokenize(row[column].replace('“', '"').replace('”', '"').replace('’',"'")))
    tokenized_text = tokenizer.tokenize(text.replace('“', '"').replace('”', '"').replace('’',"'"))
    tags = [0]*len(tokenized_text)
    for span_type in spans.keys():
        if len(span_type) != 0:
            tags = get_binary_list_absa(tokenized_text, tags, spans[span_type], span_type)

    return tags


def get_gold_labels(task_name, row, columns):
    bin_argument = None
    if task_name == "NER":
        bin_argument = get_gold_labels_ner(row, columns)
    elif task_name == "DEF":
        bin_argument = get_gold_labels_def(row, columns)
    elif task_name == "ABSA":
        bin_argument = get_gold_labels_absa(row, columns)

    return bin_argument


def get_spans(task_name, row, output):
    bin_argument = None
    if task_name == "NER":
        bin_argument = get_entities_ner(row, output)
    elif task_name == "DEF":
        bin_argument = get_entities_def(row, output)
    elif task_name == "ABSA":
        bin_argument = get_aspect_opinions_absa(row, output)
    
    return bin_argument


def get_answers(output):
    output = output[0].split(": ")
    answer = str(output[1]) if len(output) > 1 else ""
    return answer


def get_squad_example(row):
    example = f"Answer: {row['answer_text']}\n"
    return example

def get_sa_example(row):
    example = f"Sentiment Label: {row['label_text']}\n"
    return example

def get_absa_example(row, columns):
    example = ""
    for column in columns:
        if not pd.isna(row[column]) and 'Opinion' in column:
            example+=f"{column}: {row[column]}\n"
        if not pd.isna(row[column]) and 'Aspect' in column:
            example+=f"{column}: {row[column]}\n"
    return example
    
def get_ner_example(row, columns):
    example = ""
    if 'lang' in row and row['lang'] != 'en':
        # tokenizer = MosesDetokenizer(lang=row['lang'])
        tokenizer = MultilingualProcessor(row['lang'], detokenize=True)
    else:
        tokenizer = TreebankWordDetokenizer()
    tokens = ast.literal_eval(row['tokens'])
    id_label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    tag_count = {'PER': 0, 'LOC': 0, 'ORG': 0, 'MISC': 0}
    tag_to_string = {'PER': 'Person', 'LOC': 'Location', 'ORG': 'Organization', 'MISC': 'Miscellaneous Name'}
    start_id = None
    tags = ast.literal_eval(row['ner_tags'])
    for i, tag in enumerate(tags):
        if tag == 0:
            if start_id != None:
                span = tokenizer.tokenize(tokens[start_id:i]) if isinstance(tokenizer,TreebankWordDetokenizer) else tokenizer.detokenize(tokens[start_id:i])
                tag_name = id_label_map[tags[start_id]].split('-')[1]
                example+=f"{tag_to_string[tag_name]} {tag_count[tag_name]+1}: {span}\n"
                tag_count[tag_name] += 1
                start_id = None
            else:
                continue
        else:
            if start_id == None:
                start_id = i
            else:
                if tags[start_id] == tag-1:
                    continue
                else:
                    span = tokenizer.tokenize(tokens[start_id:i]) if isinstance(tokenizer,TreebankWordDetokenizer) else tokenizer.detokenize(tokens[start_id:i])
                    tag_name = id_label_map[tags[start_id]].split('-')[1]
                    example+=f"{tag_to_string[tag_name]} {tag_count[tag_name]+1}: {span}\n"
                    tag_count[tag_name] += 1
                    start_id = i
    if start_id != None:
        span = tokenizer.tokenize(tokens[start_id:]) if isinstance(tokenizer,TreebankWordDetokenizer) else tokenizer.detokenize(tokens[start_id:])
        tag_name = id_label_map[tags[start_id]].split('-')[1]
        example+=f"{tag_to_string[tag_name]} {tag_count[tag_name]+1}: {span}\n"
        tag_count[tag_name] += 1

    return example

def get_def_example(row, column):
    example = ""
    tokenizer = TreebankWordDetokenizer()
    tokens = ast.literal_eval(row['tokens'])
    id_label_map = {
        0: 'O', 1: 'B-Alias-Term', 2: 'I-Alias-Term', 3: 'B-Alias-Term-frag', 4: 'I-Alias-Term-frag', 5: 'B-Definition', 
        6: 'I-Definition', 7: 'B-Definition-frag', 8: 'I-Definition-frag', 9: 'B-Ordered-Definition', 10: 'I-Ordered-Definition', 
        11: 'B-Ordered-Term', 12: 'I-Ordered-Term', 13: 'B-Qualifier', 14: 'I-Qualifier', 15: 'B-Referential-Definition', 
        16: 'I-Referential-Definition', 17: 'B-Referential-Term', 18: 'I-Referential-Term', 19: 'B-Secondary-Definition', 
        20: 'I-Secondary-Definition', 21: 'B-Term', 22: 'I-Term', 23: 'B-Term-frag', 24: 'I-Term-frag'
    }
    tag_count = {
        'Qualifier': 0, 'Definition': 0, 'Ordered Definition': 0, 'Referential Term': 0, 'Term frag': 0, 
        'Alias Term frag': 0, 'Definition frag': 0, 'Alias Term': 0, 'Secondary Definition': 0, 'Term': 0, 
        'Referential Definition': 0, 'Ordered Term': 0
    }
    start_id = None
    tags = ast.literal_eval(row['def_tags'])
    for i, tag in enumerate(tags):
        if tag == 0:
            if start_id != None:
                span = tokenizer.tokenize(tokens[start_id:i])
                tag_name = id_label_map[tags[start_id]].split('-')
                tag_name = " ".join(tag_name[1:])
                example+=f"{tag_name} {tag_count[tag_name]+1}: {span}\n"
                tag_count[tag_name] += 1
                start_id = None
            else:
                continue
        else:
            if start_id == None:
                start_id = i
            else:
                if tags[start_id] == tag-1:
                    continue
                else:
                    span = tokenizer.tokenize(tokens[start_id:i])
                    tag_name = id_label_map[tags[start_id]].split('-')
                    tag_name = " ".join(tag_name[1:])
                    example+=f"{tag_name} {tag_count[tag_name]+1}: {span}\n"
                    tag_count[tag_name] += 1
                    start_id = i
    if start_id != None:
        span = tokenizer.tokenize(tokens[start_id:])
        tag_name = id_label_map[tags[start_id]].split('-')
        tag_name = " ".join(tag_name[1:])
        example+=f"{tag_name} {tag_count[tag_name]+1}: {span}\n"
        tag_count[tag_name] += 1

    return example