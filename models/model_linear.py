import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, vocab_size, tag_size, template_size, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.template_classifier = nn.Linear(embed_dim, template_size)
        self.token_tagger = nn.Linear(embed_dim, tag_size)

    def forward(self, input_ids):

        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        pooled = embeddings.mean(dim=1)  # [batch_size, embed_dim]

        template_logits = self.template_classifier(pooled)  # [batch_size, template_size]
        tag_logits = self.token_tagger(embeddings)  # [batch_size, seq_len, tag_size]

        return template_logits, tag_logits