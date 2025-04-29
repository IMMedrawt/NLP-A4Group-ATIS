# train_linear.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ATISDataset, build_token_vocab, build_tag_vocab
from models.model_linear import LinearClassifier
from trainer import train_epoch, evaluate
from utils import pad_batch, EarlyStopping

def main():
    # Settings
    data_path = "atis.json"
    split_type = "question-split"  # or "query-split"
    batch_size = 32
    embed_dim = 100
    lr = 1e-3
    max_len = 64
    num_epochs = 20
    patience = 3  # early stopping patience
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabularies
    token2id = build_token_vocab(data_path)
    tag2id = build_tag_vocab(data_path)

    vocab_size = len(token2id)
    tag_size = len(tag2id)
    template_size = 200  # You need to scan how many SQL templates there are

    # Create datasets
    train_dataset = ATISDataset(data_path, token2id, tag2id, split_type, mode="train", max_len=max_len)
    dev_dataset = ATISDataset(data_path, token2id, tag2id, split_type, mode="dev", max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

    # Create model
    model = LinearClassifier(vocab_size, tag_size, template_size, embed_dim)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopping(patience=patience, verbose=True)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        template_acc, tag_acc = evaluate(model, dev_loader, device)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Dev Template Acc={template_acc:.4f}, Dev Tag Acc={tag_acc:.4f}")

        # Early stopping
        early_stopper(train_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    # Save model
    torch.save(model.state_dict(), "linear_model.pth")
    print("Model saved to linear_model.pth")

if __name__ == "__main__":
    main()
