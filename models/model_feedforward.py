import torch
import torch.nn as nn

class FeedforwardClassifier(nn.Module):
    def __init__(self, vocab_size, tag_size, template_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.template_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, template_size)
        )

        self.token_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tag_size)
        )

    def forward(self, input_ids):

        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        pooled = embeddings.mean(dim=1)  # [batch_size, embed_dim]

        template_logits = self.template_ffn(pooled)  # [batch_size, template_size]
        tag_logits = self.token_ffn(embeddings)  # [batch_size, seq_len, tag_size]

        return template_logits, tag_logits