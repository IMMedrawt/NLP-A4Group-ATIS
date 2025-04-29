# vocab.py

import json
from collections import Counter

def build_token_vocab(data_path, min_freq=1, save_path="token_vocab.json"):
    """
    Build token vocabulary from atis.json
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    counter = Counter()

    for item in data:
        for sentence in item["sentences"]:
            tokens = sentence["text"].lower().split()
            counter.update(tokens)
    
    token2id = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            token2id[token] = len(token2id)
    
    with open(save_path, "w") as f:
        json.dump(token2id, f, indent=2)
    
    return token2id

def build_tag_vocab(data_path, save_path="tag_vocab.json"):
    """
    Build tag vocabulary from atis.json
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    tags = set()
    for item in data:
        for sentence in item["sentences"]:
            for var in sentence["variables"].keys():
                tags.add(var)
    
    tag2id = {"O": 0}
    for tag in sorted(tags):
        tag2id[tag] = len(tag2id)
    
    with open(save_path, "w") as f:
        json.dump(tag2id, f, indent=2)
    
    return tag2id

def load_vocab(path):
    """
    Load a vocab dictionary from json
    """
    with open(path, "r") as f:
        vocab = json.load(f)
    return vocab