import torch
import torch.nn.functional as F

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

# Evaluate model on dev set
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
