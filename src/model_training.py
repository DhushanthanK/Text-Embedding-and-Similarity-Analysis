import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        return question, answer

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, output_embed_dim):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=3
        )
        self.projection = torch.nn.Linear(embed_dim, output_embed_dim)
    
    def forward(self, tokenizer_output):
        x = self.embedding_layer(tokenizer_output['input_ids'])
        x = self.encoder(x, src_key_padding_mask=tokenizer_output['attention_mask'].logical_not())
        cls_embed = x[:,0,:]
        return self.projection(cls_embed)

def train(dataset, num_epochs=10, batch_size=32, embed_size=512, output_embed_size=128, max_seq_len=64):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)
    dataloader = DataLoader(QADataset(dataset), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(question_encoder.parameters()) + list(answer_encoder.parameters()), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = []
        for idx, (question, answer) in enumerate(dataloader):
            question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            answer_tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            question_embed = question_encoder(question_tok)
            answer_embed = answer_encoder(answer_tok)
            similarity_scores = question_embed @ answer_embed.T
            target = torch.arange(question_embed.shape[0], dtype=torch.long)
            loss = loss_fn(similarity_scores, target)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return question_encoder, answer_encoder
