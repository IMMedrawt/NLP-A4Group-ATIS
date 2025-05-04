import torch
import torch.nn as nn
import math
from sklearn.linear_model import LogisticRegression
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.feature_extraction.text import CountVectorizer

# Linear Classifier (Logistic Regression)
class LinearClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

# Feedforward Neural Network Classifier
class FeedforwardClassifier:
    def __init__(self, input_dim=None, hidden_dim=256, num_epochs=20, lr=0.001):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.lr = lr

    def train(self, X_train, y_train):
        import torch.utils.data as data
        self.label_encoder.fit(y_train)
        y_encoded = self.label_encoder.transform(y_train)
        num_classes = len(self.label_encoder.classes_)

        X_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        dataset = data.TensorDataset(X_tensor, y_tensor)
        loader = data.DataLoader(dataset, batch_size=64, shuffle=True)

        self.model = nn.Sequential(
            nn.Linear(X_tensor.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.num_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1)
        return self.label_encoder.inverse_transform(preds.numpy())

# LSTM-based Classifier
class LSTMClassifier:
    def __init__(self, vocab=None, hidden_dim=128, num_epochs=10, lr=0.001):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = CountVectorizer()
        self.word2idx = {}
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.max_len = 40

    def tokenize(self, sentence):
        return sentence.lower().split()

    def encode(self, sentences):
        encoded = []
        for sent in sentences:
            tokens = self.tokenize(sent)
            ids = [self.word2idx.get(tok, self.word2idx['<UNK>']) for tok in tokens]
            ids = ids[:self.max_len] + [self.word2idx['<PAD>']] * (self.max_len - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def train(self, X_train_texts, y_train):
        import torch.utils.data as data

        
        vocab = set(tok for sent in X_train_texts for tok in self.tokenize(sent))
        self.word2idx = {tok: idx + 2 for idx, tok in enumerate(sorted(vocab))}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1

        X_tensor = self.encode(X_train_texts)

        
        self.label_encoder.fit(y_train)
        y_tensor = torch.tensor(self.label_encoder.transform(y_train), dtype=torch.long)

        dataset = data.TensorDataset(X_tensor, y_tensor)
        loader = data.DataLoader(dataset, batch_size=64, shuffle=True)

        num_classes = len(self.label_encoder.classes_)
        vocab_size = len(self.word2idx)

        self.model = nn.Sequential(
            nn.Embedding(vocab_size, 128, padding_idx=0),
            nn.LSTM(128, self.hidden_dim, batch_first=True),
            nn.Flatten(),
            nn.Linear(self.hidden_dim * self.max_len, num_classes)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.num_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs, _ = self.model[1](self.model[0](X_batch))  # unpack lstm
                flat = self.model[2](outputs)
                logits = self.model[3](flat)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X_test_texts):
        self.model.eval()
        X_tensor = self.encode(X_test_texts)
        with torch.no_grad():
            outputs, _ = self.model[1](self.model[0](X_tensor))
            flat = self.model[2](outputs)
            logits = self.model[3](flat)
            preds = torch.argmax(logits, dim=1)
        return self.label_encoder.inverse_transform(preds.numpy())

# Transformer-based Classifier (BERT)
def sinusoidal_position_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, max_len, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.positional = nn.Parameter(sinusoidal_position_encoding(max_len, embed_dim), requires_grad=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        mask = (x == self.embedding.padding_idx)  # [B, L]
        x = self.embedding(x) + self.positional[:, :x.size(1), :]  # [B, L, D]
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, L, D]
        x = x.permute(0, 2, 1)  # [B, D, L]
        x = self.pool(x).squeeze(2)  # [B, D]
        return self.classifier(x)

class TransformerTextClassifier:
    def __init__(self, max_len=40, embed_dim=64, num_heads=4, hidden_dim=128, num_epochs=20, lr=0.001):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.word2idx = {}
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.lr = lr

    def tokenize(self, sentence):
        return sentence.lower().split()

    def encode(self, sentences):
        encoded = []
        for sent in sentences:
            tokens = self.tokenize(sent)
            ids = [self.word2idx.get(tok, self.word2idx['<UNK>']) for tok in tokens]
            ids = ids[:self.max_len] + [self.word2idx['<PAD>']] * (self.max_len - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def train(self, X_train_texts, y_train):
        import torch.utils.data as data

        vocab = set(tok for sent in X_train_texts for tok in self.tokenize(sent))
        self.word2idx = {tok: idx + 2 for idx, tok in enumerate(sorted(vocab))}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        pad_idx = self.word2idx['<PAD>']

        X_tensor = self.encode(X_train_texts)

        self.label_encoder.fit(y_train)
        y_tensor = torch.tensor(self.label_encoder.transform(y_train), dtype=torch.long)

        dataset = data.TensorDataset(X_tensor, y_tensor)
        loader = data.DataLoader(dataset, batch_size=64, shuffle=True)

        num_classes = len(self.label_encoder.classes_)
        vocab_size = len(self.word2idx)

        self.model = TransformerClassifier(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            max_len=self.max_len,
            pad_idx=pad_idx
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.num_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X_test_texts):
        self.model.eval()
        X_tensor = self.encode(X_test_texts)
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1)
        return self.label_encoder.inverse_transform(preds.numpy())