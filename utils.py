# utils.py

import torch

def pad_batch(batch, pad_token_id=0):
    """
    Pad input_ids and tag_ids in a batch to the maximum length in that batch.
    Args:
        batch (list of dict): List of samples from Dataset
    Returns:
        batch dict with tensors
    """
    max_len = max(len(sample['input_ids']) for sample in batch)

    input_ids = []
    tag_ids = []
    template_labels = []
    variables = []
    tokens = []

    for sample in batch:
        input_id = sample['input_ids']
        tag_id = sample['tag_ids']
        
        pad_len = max_len - len(input_id)
        input_ids.append(torch.cat([input_id, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        tag_ids.append(torch.cat([tag_id, torch.full((pad_len,), 0, dtype=torch.long)]))  # O label id is 0

        template_labels.append(sample.get('template_id', 0))  # if not available, use dummy 0
        variables.append(sample['variables'])
        tokens.append(sample['tokens'])

    return {
        'input_ids': torch.stack(input_ids),
        'tag_ids': torch.stack(tag_ids),
        'template_labels': torch.tensor(template_labels, dtype=torch.long),
        'variables': variables,
        'tokens': tokens
    }

def calculate_accuracy(preds, labels):
    """
    Calculate simple accuracy
    Args:
        preds: predicted labels
        labels: ground truth labels
    """
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total

class EarlyStopping:
    """
    Early stopping to stop training when validation loss does not improve.
    """
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
