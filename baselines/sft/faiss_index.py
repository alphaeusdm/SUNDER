import json
import os
import numpy as np
import pandas as pd
import torch
import faiss
# from faiss import write_index, read_index
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional, Any
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm


class FaissIndex:
    def __init__(
        self,
        task_name: str = "TBO",
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
    ):
        self.task_name = task_name
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        
    def setup(self) -> None:
        # Get tokenizer
        self.tokenizer = self.get_tokenizer()
        # Get embedding model
        self.embedding_model = self.get_model()
        # Build and get index
        dataset_dir = f'embeddings/{self.task_name}/'
        index_fname = f'{dataset_dir}{self.task_name}.index'
        # if not os.path.isfile(index_fname):
        self.build_index(self.task_name, dataset_dir)
        # self.index = read_index(index_fname) # Path of index

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name_or_path)

    def get_model(self):
        return SentenceTransformer(self.model_name_or_path,device="cuda")

    def get_embeddings(self, sentences):
        return self.embedding_model.encode(sentences,device="cuda")

    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        mean_pool_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return mean_pool_embeddings
        
    def build_index(self, dataset_name, dataset_dir) -> None:
        sentences = pd.read_csv(f'data/{dataset_name}/train.csv', sep='\t')['text'].tolist() if dataset_name != "SQUAD" \
            else pd.read_csv(f'data/{dataset_name}/train.csv', sep='\t')['question'].tolist()
        sentence_embeddings = self.get_embeddings(sentences)
        d = sentence_embeddings.shape[1]
        ngpus = faiss.get_num_gpus()
        self.index = faiss.IndexFlatL2(d)
        self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(sentence_embeddings)
        # Save index for future use
        # write_index(index, f'{dataset_dir}{dataset_name}.index')

    def search_relevant(self, text, k=3):
        # xq = self.embedding_model.encode([query])
        xq = self.get_embeddings([text])
        D, I = self.index.search(xq, k)
        indices = I.tolist()[0]
        return indices

if __name__ == '__main__':
    _ = FaissIndex()
