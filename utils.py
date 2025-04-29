import torch

# Pad batch to maximum sequence length
def pad_batch(batch, pad_token_id=0):
    max_len = max(len(s['input_ids']) for s in batch)
    input_ids, tag_ids, template_labels, variables, tokens = [], [], [], [], []
    for s in batch:
        pad_len = max_len - len(s['input_ids'])
        input_ids.append(torch.cat([s['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        tag_ids.append(torch.cat([s['tag_ids'], torch.full((pad_len,), 0, dtype=torch.long)]))
        template_labels.append(s.get('template_id', 0))
        variables.append(s['variables'])
        tokens.append(s['tokens'])
    return {
        'input_ids': torch.stack(input_ids),
        'tag_ids': torch.stack(tag_ids),
        'template_labels': torch.tensor(template_labels, dtype=torch.long),
        'variables': variables,
        'tokens': tokens
    }

# Calculate prediction accuracy
def calculate_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total

# Early stopping callback class
class EarlyStopping:
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
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
