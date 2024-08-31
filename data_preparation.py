# data_preparation.py

import os
import tiktoken
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class TextDataset(Dataset):
    def __init__(self, tokenized_texts, vocab_size):
        self.tokenized_texts = tokenized_texts
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.tokenized_texts[idx], dtype=torch.long)}

def tokenize_texts(texts, tokenizer, max_length=20):
    tokenized_texts = []
    for text in texts:
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length")
        tokenized_texts.append(tokens)
    return tokenized_texts

def prepare_data(cache_file="cached_tokenized_data.pkl"):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            tokenized_texts, vocab_size = pickle.load(f)
        print("Loaded tokenized data from cache.")
    else:
        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = dataset['text']

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = tokenizer.n_vocab
        
        # Tokenize the dataset
        tokenized_texts = tokenize_texts(texts, tokenizer, max_length=20)
        
        # Cache the tokenized data
        with open(cache_file, 'wb') as f:
            pickle.dump((tokenized_texts, vocab_size), f)
        print("Tokenized data cached.")

    return TextDataset(tokenized_texts, vocab_size)

def get_dataloader(batch_size=32, shuffle=True):
    dataset = prepare_data()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

