import re
import numpy as np
import nltk
import evaluate
from evaluate import load
from nltk.corpus import stopwords
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score


def get_seqeval_metrics(gold, pred):
    seqeval = evaluate.load('seqeval')
    return seqeval.compute(predictions=pred, references=gold)

def get_token_f1(gold, pred, micro=False):
    return round(np.mean([f1_score(g, p, average='macro' if not micro else 'micro', zero_division=0) for g, p in list(zip(gold, pred))]), 3)

def get_span_metrics(gold, pred, dict_idx2lbl):
    pred_lb = [[dict_idx2lbl[p] for p in pr] for pr in pred]
    gold_lb = [[dict_idx2lbl[l] for l in gl] for gl in gold]
        
    return get_token_f1(gold, pred), round(get_seqeval_metrics(gold_lb, pred_lb)['overall_f1'], 3)

def get_f1_score(gold, pred):
    return round(f1_score(gold, pred, average='macro'), 3)

# MHQA Metrics
def postprocess_text(text):
    # Convert to lowercase
    text = text.lower() if isinstance(text,str) else 'none'
    text = text.strip()
    # Remove formatting (punctuation, special characters)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def calculate_f1(candidate, reference):
    reference_tokens = set(nltk.word_tokenize(reference))
    candidate_tokens = set(nltk.word_tokenize(candidate))
    common_tokens = reference_tokens.intersection(candidate_tokens)
    if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
        return 0.0
    precision = len(common_tokens) / len(candidate_tokens)
    recall = len(common_tokens) / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Function to calculate exact match
def calculate_exact_match(candidate, reference):
    return 1 if reference == candidate else 0

def calculate_bleu(candidate, reference):
    reference = [nltk.word_tokenize(reference)]
    candidate = nltk.word_tokenize(candidate)
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing_function)

def get_mhqa_metrics(gold, pred):
    test_samples = [(postprocess_text(p), postprocess_text(g)) for g, p in zip(gold, pred)]
    em_scores = [calculate_exact_match(pred, gt) for pred, gt in test_samples]
    f1_scores = [calculate_f1(pred, gt) for pred, gt in test_samples]
    bleu_scores = [calculate_bleu(pred, gt) for pred, gt in test_samples]
    return {
        "em": round(sum(em_scores) / len(em_scores), 3),
        "f1": round(sum(f1_scores) / len(f1_scores), 3),
        "bleu": round(sum(bleu_scores) / len(bleu_scores), 3)
    }

