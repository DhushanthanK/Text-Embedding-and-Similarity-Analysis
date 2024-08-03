import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from embedding_utils import BertEmbeddings
from similarity import calculate_similarity

sentence1 = "The bat flew out of the cave at night."
sentence2 = "He swung the bat and hit a home run."

bert_embeddings = BertEmbeddings()
bert_embedding1 = bert_embeddings.get_embeddings(sentence1, "bat").detach().numpy()
bert_embedding2 = bert_embeddings.get_embeddings(sentence2, "bat").detach().numpy()

similarity = calculate_similarity(bert_embedding1, bert_embedding2)
print(f"Cosine Similarity between BERT embeddings in different contexts: {similarity}")
