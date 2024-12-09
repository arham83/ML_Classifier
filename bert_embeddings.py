import torch
import numpy as np
from transformers import BertTokenizer, BertModel

class BertEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_bert_embedding(self, doc, embedding_strategy='mean'):
        inputs = self.tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        if embedding_strategy == 'cls':
            embedding = outputs.last_hidden_state[:, 0, :]
        elif embedding_strategy == 'mean':
            attention_mask = inputs['attention_mask']
            embedding = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        elif embedding_strategy == 'max':
            embedding = torch.max(outputs.last_hidden_state, dim=1)[0]
        else:
            raise ValueError(f"Unsupported embedding strategy: {embedding_strategy}")

        return embedding.squeeze().numpy()
