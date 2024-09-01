import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_texts = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )['input_ids'].tolist()
    return tokenized_texts

def encode_text(tokenizer, text):
    return tokenizer.encode(text, return_tensors='pt').tolist()

def decode_text(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)

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
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        vocab_size = len(tokenizer)

        # Tokenize the dataset
        tokenized_texts = tokenize_texts(texts, tokenizer, max_length=20)
        
        # Cache the tokenized data
        with open(cache_file, 'wb') as f:
            pickle.dump((tokenized_texts, vocab_size), f)
        print("Tokenized data cached.")

    return TextDataset(tokenized_texts, vocab_size)

def prepare_validation_test_data():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = dataset['text']
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_texts = tokenize_texts(texts, tokenizer, max_length=20)
    val_dataset = TextDataset(tokenized_texts, len(tokenizer))
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = dataset['text']
    tokenized_texts = tokenize_texts(texts, tokenizer, max_length=20)
    test_dataset = TextDataset(tokenized_texts, len(tokenizer))

    return val_dataset, test_dataset

def get_dataloader(batch_size=32, shuffle=True):
    dataset = prepare_data()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    # Example usage of encode and decode functions
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sample_text = "Hello, how are you?"
    encoded = encode_text(tokenizer, sample_text)
    decoded = decode_text(tokenizer, encoded[0])  # Decode the first item in the batch
    print("Encoded:", encoded)
    print("Decoded:", decoded)

    # Prepare validation and test datasets
    val_dataset, test_dataset = prepare_validation_test_data()
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)