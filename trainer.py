import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
import torch.nn.functional as F
import torch.optim as optim

class ATISDataset(Dataset):
    def __init__(self, data_path, token2id, tag2id, split_type='question-split', mode='train', max_len=128):
        """
        Load the ATIS dataset.
        Args:
            data_path (str): Path to the atis.json file.
            token2id (dict): Mapping from tokens to their IDs.
            tag2id (dict): Mapping from tags to their IDs.
            split_type (str): 'question-split' or 'query-split'.
            mode (str): 'train', 'dev', or 'test'.
            max_len (int): Maximum sequence length.
        """
        self.token2id = token2id
        self.tag2id = tag2id
        self.max_len = max_len

        # Load the raw json data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        # Process the data to create samples
        self.samples = self.build_samples(raw_data, split_type, mode)

    def build_samples(self, raw_data, split_type, mode):
        """
        Build the sample list according to split type and mode.
        Each sample contains: tokenized text, token labels, template id, variable values, raw SQL.
        """
        samples = []

        for item in raw_data:
            if 'sentences' not in item:
                continue

            for sent in item['sentences']:
                if sent[split_type] != mode:
                    continue

                tokens = sent['text'].split()  # simple whitespace tokenization
                variables = sent['variables']

                # Prepare tagging labels for each token
                tags = self.create_tags(tokens, variables)

                # Pick the shortest SQL template from sql list
                sql_templates = item['sql']
                sql_template = min(sql_templates, key=lambda x: (len(x), x))

                samples.append({
                    'tokens': tokens,
                    'tags': tags,
                    'template': sql_template,
                    'variables': variables
                })

        return samples

    def create_tags(self, tokens, variables):
        """
        Create tag list for each token.
        Default tag is 'O' (outside).
        If a token matches a variable value, tag it with the variable name.
        """
        tags = ['O'] * len(tokens)

        # Simple matching: exact match token to variable value
        for var_name, var_value in variables.items():
            var_tokens = var_value.split()
            for i in range(len(tokens) - len(var_tokens) + 1):
                if tokens[i:i+len(var_tokens)] == var_tokens:
                    for j in range(len(var_tokens)):
                        tags[i+j] = var_name

        return tags

    def __len__(self):
        """
        Return the number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get one sample.
        Return input_ids, tag_ids, template_text, variables.
        """
        sample = self.samples[idx]
        tokens = sample['tokens']
        tags = sample['tags']
        template = sample['template']
        variables = sample['variables']

        # Encode tokens and tags
        input_ids = [self.token2id.get(t, self.token2id['<UNK>']) for t in tokens]
        tag_ids = [self.tag2id.get(tag, self.tag2id['O']) for tag in tags]

        # Pad or truncate
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            tag_ids = tag_ids[:self.max_len]
        else:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.token2id['<PAD>']] * pad_len
            tag_ids += [self.tag2id['O']] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'tag_ids': torch.tensor(tag_ids, dtype=torch.long),
            'template': template,  # return template text for template prediction task
            'variables': variables, # keep variables for later use in SQL reconstruction
            'tokens': tokens        # optionally keep original tokens
        }

# Utility functions for building vocabularies

def build_token_vocab(data_path, min_freq=1):
    """
    Build token vocabulary from training data.
    Args:
        data_path (str): Path to atis.json.
        min_freq (int): Minimum frequency to include a token.
    Returns:
        token2id (dict): Token to ID mapping.
    """
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    counter = Counter()
    for item in raw_data:
        for sent in item['sentences']:
            tokens = sent['text'].split()
            counter.update(tokens)

    token2id = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            token2id[token] = len(token2id)

    return token2id

def build_tag_vocab(data_path):
    """
    Build tag vocabulary from training data.
    Args:
        data_path (str): Path to atis.json.
    Returns:
        tag2id (dict): Tag to ID mapping.
    """
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    tag_set = set()
    for item in raw_data:
        for sent in item['sentences']:
            variables = sent['variables']
            for var_name in variables.keys():
                tag_set.add(var_name)

    tag2id = {'O': 0}
    for tag in sorted(tag_set):
        tag2id[tag] = len(tag2id)

    return tag2id

# Training and evaluation functions
def train_epoch(model, dataloader, optimizer, device):
    """
    Train one epoch
    """
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        tag_ids = batch['tag_ids'].to(device)

        optimizer.zero_grad()
        template_logits, tag_logits = model(input_ids)

        template_labels = batch['template_labels'].to(device)
        loss_template = F.cross_entropy(template_logits, template_labels)

        loss_tags = F.cross_entropy(tag_logits.view(-1, tag_logits.size(-1)), tag_ids.view(-1))

        loss = loss_template + loss_tags
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    """
    Evaluate on validation set
    """
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

            mask = (input_ids != 0)  # mask out padding
            correct_tags += ((tag_preds == tag_ids) * mask).sum().item()
            total_tags += mask.sum().item()

    template_acc = correct_template / total_template
    tag_acc = correct_tags / total_tags

    return template_acc, tag_acc
