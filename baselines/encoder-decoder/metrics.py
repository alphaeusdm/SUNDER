import re
import numpy as np
import evaluate
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score


def get_ner_metrics(gold, pred):
    seqeval = evaluate.load('seqeval')
    return seqeval.compute(predictions=pred, references=gold)

def get_token_f1(gold, pred, micro=False):
    return round(np.mean([f1_score(g, p, average='macro' if not micro else 'micro', zero_division=0) for g, p in list(zip(gold, pred))]), 3)

def get_span_f1(gold, pred):
    total_gold = 0
    total_pred = 0
    correct = 0

    for g, p in list(zip(gold, pred)):
        gold_spans = []
        pred_spans = []
        g = "".join(str(i) for i in g)
        gold_spans = [[m.start(), m.end()-1] for m in re.finditer('12*|2+',g)]
        p = "".join(str(i) for i in p)
        pred_spans = [[m.start(), m.end()-1] for m in re.finditer('12*|2+',p)]
        correct += len([item for item in pred_spans if item in gold_spans])
        total_gold += len(gold_spans)
        total_pred += len(pred_spans)

    precision = correct/total_pred if total_pred > 0 else 0
    recall = correct/total_gold

    if precision == 0 or recall == 0:
        # print(correct, total_gold, total_pred)
        return 0
    
    f1 = 2*precision*recall/(precision + recall)

    return round(f1, 3)

def get_metrics(gold, pred):
    return get_token_f1(gold, pred), get_span_f1(list(map(lambda x: 1 if x==2 else x, gold)),list(map(lambda x: 1 if x==2 else x,pred)))