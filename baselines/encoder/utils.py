'''
part of the code reused from x-claim repository: https://github.com/mbzuai-nlp/x-claim
'''

import json
import ast
import ipdb
import pandas as pd
from nltk .tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import stanza
from typing import List, Optional


class MultilingualProcessor:    
    def __init__(self, lang: Optional[str] = None):
        self.lang = lang
        self.nlp = None
        
        if lang:
            try:
                self.nlp = stanza.Pipeline(
                    lang, 
                    processors='tokenize',
                    verbose=False,
                    download_method=None
                )
            except:
                print(f"✗ Model not found for '{lang}'")
                print(f"  Download with: stanza.download('{lang}')")
                raise
    
    def tokenize(self, text: str) -> List[str]:
        if not self.nlp:
            raise ValueError("No language model loaded. Initialize with a language code.")
        
        doc = self.nlp(text)
        tokens = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                tokens.append(token.text)
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        
        text = ""
        for i, token in enumerate(tokens):
            text += token
            
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                if self._needs_space(token, next_token):
                    text += " "
        
        return text
    
    def _needs_space(self, current: str, next_token: str) -> bool:
        if next_token in '.,!?;:)]}»"\'':
            return False
        
        if current in '([{«"\'':
            return False
        
        if next_token.startswith("'") or next_token.startswith("'"):
            return False
        
        if current == '-' or next_token == '-':
            return False
        
        return True



def find_sublist_indices(main_list, sublist):
    start_index = -1
    end_index = -1
    for i in range(len(main_list) - len(sublist) + 1):
        if main_list[i:i + len(sublist)] == sublist:
            start_index = i
            end_index = i + len(sublist) - 1
            break

    return start_index, end_index

def process_data(data_file):
    data = pd.read_csv(data_file, sep='\t')
    data_new = pd.DataFrame()

    if 'lang' in data.columns:
        print(data.lang.unique()[0])
        stanza.download(data.lang.unique()[0])
        # tokenizer = MosesTokenizer(lang=row['lang'])
        tokenizer = MultilingualProcessor(lang=data.lang.unique()[0])
    else:
        tokenizer = TreebankWordTokenizer()

    for ind, row in data.iterrows():
        row = row.dropna()
        text = tokenizer.tokenize(row['text'])
        start_type, end_type = list(), list()
        for column in row.index.to_list()[1:]:
            if "Argument" in column or "span" in column:
                span = tokenizer.tokenize(row[column])
                start_index, end_index = find_sublist_indices(text, span)
                start_type.append(start_index)
                end_type.append(end_index)
        new_row = pd.DataFrame([{'tokens': text, 'span_type_start_index': start_type, 'span_type_end_index': end_type}])
        data_new = pd.concat([data_new, new_row], ignore_index=True)
    
    return data_new


def get_binary_list_absa(tokenized_text, tags, spans, span_type):
    label_id_map = {'O': 0, 'B-Aspect': 1, 'I-Aspect': 2, 'B-Opinion': 3, 'I-Opinion': 4}
    for span in spans:
        span_len = len(span)
        for i in range(len(tokenized_text)-span_len+1):
            if tokenized_text[i:i+span_len]==span and tags[i:i+span_len]==[0]*span_len:
                tags[i:i+span_len] = [label_id_map[f'B-{span_type}']]+[label_id_map[f'I-{span_type}']]*(span_len-1)
    return tags


def get_binary_labels(seq, row, start_index, end_index):   
    text = row['text']
    tokenizer = TreebankWordTokenizer()
    id_label_map = {0: 'O', 1: 'B-Aspect', 2: 'I-Aspect', 3: 'B-Opinion', 4: 'I-Opinion'}
    spans = {'Aspect': [], 'Opinion': []}
    for column, val in row.items():
        if 'aspect' in column.lower() and not pd.isna(row[column]):
            spans['Aspect'].append(tokenizer.tokenize(row[column].replace('“', '"').replace('”', '"').replace('’',"'")))
        elif 'opinion' in column.lower() and not pd.isna(row[column]):
            spans['Opinion'].append(tokenizer.tokenize(row[column].replace('“', '"').replace('”', '"').replace('’',"'")))
    tokenized_text = tokenizer.tokenize(text.replace('“', '"').replace('”', '"').replace('’',"'"))
    tags = [0]*len(tokenized_text)
    for span_type in spans.keys():
        if len(span_type) != 0:
            tags = get_binary_list_absa(tokenized_text, tags, spans[span_type], span_type)
    tags = [id_label_map[tag] for tag in tags]

    return tags


def label_update(y_true, y_pred):
    new_y_true, new_y_pred = [], []
    for (y1, y2) in zip(y_true, y_pred):
        # filter out all -100 positions (special tokens, subwords, padding)
        filtered_true, filtered_pred = [], []
        for label, pred in zip(y1, y2):
            if label != -100:
                filtered_true.append(label)
                filtered_pred.append(pred)
        new_y_true.append(filtered_true)
        new_y_pred.append(filtered_pred)
    return new_y_true, new_y_pred

def get_token_labels(labels, word_ids_all, dict_lbl2idx, label_all_tokens):
    labels_tokenized = []
    for idx, seq_label in enumerate(labels):
        word_ids = word_ids_all[idx]
        previous_word_idx = None
        seq_label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
            if word_idx is None: # tokenizer.special_tokens_map
                seq_label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                text_label = seq_label[word_idx]
                seq_label_ids.append(dict_lbl2idx[text_label])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx == previous_word_idx:
                text_label = seq_label[word_idx]
                seq_label_ids.append(dict_lbl2idx[text_label] if label_all_tokens else -100)
            previous_word_idx = word_idx
        
        assert len(word_ids) == len(seq_label_ids)
        labels_tokenized.append(seq_label_ids)

    return labels_tokenized

def get_word_labels(testset, subword_labels, subword_preds):
    word_labels, word_preds = [], []
    for idx, (label, pred) in enumerate(zip(subword_labels, subword_preds)):
        word_ids = testset.word_ids[idx]
        label, pred, word_ids = label[1:], pred[1:], word_ids[1:]
        assert len(word_ids) == len(pred) and len(pred) == len(label), ipdb.set_trace()

        # get word level labels (in gold and pred) from subword piece level
        word_pred, word_label = [], []
        previous_word_idx = None
        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                break
            elif word_idx != previous_word_idx:
                word_label.append(label[subword_idx])
                word_pred.append(pred[subword_idx])
            
            previous_word_idx = word_idx
        
        if len(word_pred) != len(word_label):
            print('possibly found where white space is causing tokenization mismap issue')
            seq = testset.seqs[idx]
            seq = [word for word in seq if len(word)!=0]
            assert len(word_label)==len(seq), ipdb.set_trace()

        if len(word_label) != len(testset.seqs[idx]):
            word_label = []
            for tag in testset.examples[1][idx]:
                if tag == 'O':
                    word_label.append(0)
                else:
                    word_label.append(1)
            word_pred = word_pred + [0] * (len(word_label) - len(word_pred))
        
        word_labels.append(word_label) 
        word_preds.append(word_pred)
    
    return word_preds, word_labels