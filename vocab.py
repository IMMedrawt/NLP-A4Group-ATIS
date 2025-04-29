import json
from collections import Counter

# Build token-to-ID mapping from dataset
def build_token_vocab(data_path, min_freq=1, save_path=None):
    with open(data_path, "r") as f:
        data = json.load(f)

    counter = Counter()
    for item in data:
        for sent in item.get("sentences", []):
            tokens = sent["text"].split()
            counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(vocab, f, indent=2)

    return vocab

# Build tag-to-ID mapping from dataset
def build_tag_vocab(data_path, save_path=None):
    with open(data_path, "r") as f:
        data = json.load(f)

    tag_set = set()
    for item in data:
        for sent in item.get("sentences", []):
            tag_set.update(sent.get("variables", {}).keys())

    tag2id = {"O": 0}
    for tag in sorted(tag_set):
        tag2id[tag] = len(tag2id)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(tag2id, f, indent=2)

    return tag2id

# Load vocab from file
def load_vocab(path):
    with open(path, "r") as f:
        return json.load(f)
