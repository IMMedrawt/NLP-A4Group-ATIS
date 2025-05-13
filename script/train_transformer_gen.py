import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import TransformerSeq2Seq
from data_utils_gen import make_dataloaders, SPECIAL_TOKENS

def train_one_epoch(model, dataloader, optimizer, criterion, device, tgt_pad_idx):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=100):
    model.eval()
    device = next(model.parameters()).device
    src = src.unsqueeze(0).to(device)
    ys = torch.tensor([[tgt_vocab[SPECIAL_TOKENS['BOS']]]], dtype=torch.long).to(device)

    for _ in range(max_len):
        logits = model(src, ys)
        next_token = logits[:, -1, :].argmax(-1).unsqueeze(1)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == tgt_vocab[SPECIAL_TOKENS['EOS']]:
            break
    return ys.squeeze(0).tolist()

def decode_ids(ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = [inv_vocab.get(i, '<UNK>') for i in ids]
    if '<EOS>' in tokens:
        tokens = tokens[:tokens.index('<EOS>')]
    return ' '.join(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['geography', 'atis'])
    parser.add_argument('--split', required=True, choices=['question_split', 'query_split'])
    args = parser.parse_args()

    data_dir = f"../processed_data/{args.dataset}"
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(data_dir, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        nhead=4,
        num_layers=3,
        max_len=100,
        pad_idx=src_vocab[SPECIAL_TOKENS['PAD']]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab[SPECIAL_TOKENS['PAD']])

    for epoch in range(1, 21):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, tgt_pad_idx=tgt_vocab[SPECIAL_TOKENS['PAD']])
        print(f"[Epoch {epoch}] Train Loss: {loss:.4f}")

    # Evaluate on test set
    print("\n[Evaluating Accuracy on Test Set]")
    model.eval()
    correct = 0
    total = 0

    for src_batch, tgt_batch in test_loader:
        for src_ids, tgt_ids in zip(src_batch, tgt_batch):
            pred_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab)
            pred_tokens = decode_ids(pred_ids, tgt_vocab).split()
            gold_tokens = decode_ids(tgt_ids.tolist(), tgt_vocab).split()
            if pred_tokens == gold_tokens:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"[Accuracy] Exact-match accuracy on test set: {accuracy:.4f}")
    

if __name__ == '__main__':
    main()
