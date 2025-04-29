import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
import torch.nn.functional as F
import torch.optim as optim
from vocab import build_token_vocab, build_tag_vocab

# Dataset class for ATIS data
class ATISDataset(Dataset):
    def __init__(self, data_path, token2id, tag2id, split_type='question-split', mode='train', max_len=128):
        self.token2id = token2id
        self.tag2id = tag2id
        self.max_len = max_len
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        self.samples = self._build_samples(raw_data, split_type, mode)

    # Create usable samples with tokens, tags, and template
    def _build_samples(self, data, split_type, mode):
        samples = []
        for item in data:
            if 'sentences' not in item:
                continue
            sql_template = min(item['sql'], key=lambda x: (len(x), x))
            for sent in item['sentences']:
                if sent.get(split_type) != mode:
                    continue
                tokens = sent['text'].split()
                tags = self._create_tags(tokens, sent['variables'])
                samples.append({
                    'tokens': tokens,
                    'tags': tags,
                    'template': sql_template,
                    'variables': sent['variables']
                })
        return samples

    # Tag tokens with variable names or 'O'
    def _create_tags(self, tokens, variables):
        tags = ['O'] * len(tokens)
        for var, val in variables.items():
            val_tokens = val.split()
            for i in range(len(tokens) - len(val_tokens) + 1):
                if tokens[i:i+len(val_tokens)] == val_tokens:
                    for j in range(len(val_tokens)):
                        tags[i+j] = var
        return tags

    def __len__(self):
        return len(self.samples)

    # Convert token/tag to ID and pad
    def __getitem__(self, idx):
        s = self.samples[idx]
        input_ids = [self.token2id.get(t, self.token2id['<UNK>']) for t in s['tokens']]
        tag_ids = [self.tag2id.get(tag, self.tag2id['O']) for tag in s['tags']]
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.token2id['<PAD>']] * pad_len
            tag_ids += [self.tag2id['O']] * pad_len
        else:
            input_ids = input_ids[:self.max_len]
            tag_ids = tag_ids[:self.max_len]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'tag_ids': torch.tensor(tag_ids, dtype=torch.long),
            'template': s['template'],
            'variables': s['variables'],
            'tokens': s['tokens']
        }

# Build token-to-ID mapping
def build_token_vocab(path, min_freq=1):
    with open(path, 'r') as f:
        data = json.load(f)
    counter = Counter()
    for item in data:
        for sent in item.get('sentences', []):
            counter.update(sent['text'].split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# Build tag-to-ID mapping
def build_tag_vocab(path):
    with open(path, 'r') as f:
        data = json.load(f)
    tag_set = set()
    for item in data:
        for sent in item.get('sentences', []):
            tag_set.update(sent['variables'].keys())
    tag2id = {'O': 0}
    for tag in sorted(tag_set):
        tag2id[tag] = len(tag2id)
    return tag2id

# Train model for one epoch
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        tag_ids = batch['tag_ids'].to(device)
        optimizer.zero_grad()
        template_logits, tag_logits = model(input_ids)
        loss_template = F.cross_entropy(template_logits, batch['template_labels'].to(device))
        loss_tags = F.cross_entropy(tag_logits.view(-1, tag_logits.size(-1)), tag_ids.view(-1))
        loss = loss_template + loss_tags
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluate model on validation set
def evaluate(model, dataloader, device):
    model.eval()
    correct_template = 0
    total_template = 0
    correct_tags = 0
    total_tags = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            tag_ids = batch['tag_ids'].to(device)
            template_logits, tag_logits = model(input_ids)
            template_preds = template_logits.argmax(dim=-1)
            tag_preds = tag_logits.argmax(dim=-1)
            correct_template += (template_preds == batch['template_labels'].to(device)).sum().item()
            total_template += template_preds.size(0)
            mask = (input_ids != 0)
            correct_tags += ((tag_preds == tag_ids) * mask).sum().item()
            total_tags += mask.sum().item()
    return correct_template / total_template, correct_tags / total_tags

# test
"""
if __name__ == "__main__":
    # Example usage and test
    token2id = build_token_vocab("atis.json")
    tag2id = build_tag_vocab("atis.json")

    dataset = ATISDataset(
        data_path="atis.json",
        token2id=token2id,
        tag2id=tag2id,
        split_type='question-split',
        mode='train',
        max_len=32
    )

    print(f"Number of samples: {len(dataset)}")
    sample = dataset[0]
    print("Sample input_ids:", sample['input_ids'])
    print("Sample tag_ids:", sample['tag_ids'])
    print("Sample template:", sample['template'])
    print("Sample variables:", sample['variables'])
    print("Sample tokens:", sample['tokens'])
"""