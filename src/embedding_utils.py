import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class BertEmbeddings:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embeddings(self, sentence, word):
        inputs = self.tokenizer(sentence, return_tensors='pt')
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        word_tokens = self.tokenizer.tokenize(sentence)
        word_index = word_tokens.index(word)
        word_embedding = last_hidden_states[0, word_index + 1, :]  # +1 to account for [CLS] token
        return word_embedding

def cosine_similarity_matrix(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / norms
    similarity_matrix = np.inner(normalized_features, normalized_features)
    rounded_similarity_matrix = np.round(similarity_matrix, 4)
    return rounded_similarity_matrix

