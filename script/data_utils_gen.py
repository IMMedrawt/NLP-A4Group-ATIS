import json
import os
import spacy
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

nlp = spacy.load("en_core_web_sm")

SPECIAL_TOKENS = {
    'PAD': '<PAD>',
    'BOS': '<BOS>',
    'EOS': '<EOS>',
    'UNK': '<UNK>'
}

def tokenize(text):
    return [tok.text.lower() for tok in nlp(text)]

def build_vocab(token_lists, min_freq=1, specials=SPECIAL_TOKENS.values()):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {tok: idx for idx, tok in enumerate(specials)}
    for tok, count in counter.items():
        if count >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def encode(tokens, vocab, max_len, add_special=True):
    ids = [vocab.get(tok, vocab[SPECIAL_TOKENS['UNK']]) for tok in tokens]
    if add_special:
        ids = [vocab[SPECIAL_TOKENS['BOS']]] + ids + [vocab[SPECIAL_TOKENS['EOS']]]
    ids = ids[:max_len]
    pad_len = max_len - len(ids)
    return ids + [vocab[SPECIAL_TOKENS['PAD']]] * pad_len

class Seq2SeqDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, src_max_len=40, tgt_max_len=100):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_tokens = tokenize(src_text)
        tgt_tokens = tokenize(tgt_text)
        src_ids = encode(src_tokens, self.src_vocab, self.src_max_len, add_special=False)
        tgt_ids = encode(tgt_tokens, self.tgt_vocab, self.tgt_max_len, add_special=True)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def load_pairs(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(item['question'], item['sql']) for item in data]

def make_dataloaders(data_dir, split_type, batch_size=32, src_max_len=40, tgt_max_len=100):
    train_pairs = load_pairs(os.path.join(data_dir, f"train_{split_type}.json"))
    dev_pairs = load_pairs(os.path.join(data_dir, f"dev_{split_type}.json"))
    test_pairs = load_pairs(os.path.join(data_dir, f"test_{split_type}.json"))

    # Token lists for vocab building
    all_src = [tokenize(src) for src, _ in train_pairs]
    all_tgt = [tokenize(tgt) for _, tgt in train_pairs]
    src_vocab = build_vocab(all_src)
    tgt_vocab = build_vocab(all_tgt)

    train_ds = Seq2SeqDataset(train_pairs, src_vocab, tgt_vocab, src_max_len, tgt_max_len)
    dev_ds = Seq2SeqDataset(dev_pairs, src_vocab, tgt_vocab, src_max_len, tgt_max_len)
    test_ds = Seq2SeqDataset(test_pairs, src_vocab, tgt_vocab, src_max_len, tgt_max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, dev_loader, test_loader, src_vocab, tgt_vocab
