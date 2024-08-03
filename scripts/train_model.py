import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import load_data
from model_training import train

# Define the dataset path
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'nq_sample.tsv'))

# Load the dataset
data = load_data(data_path)

# Convert data to the format required by the model
dataset = [(row['question'], row['answer']) for _, row in data.iterrows()]

# Train the model
question_encoder, answer_encoder = train(dataset, num_epochs=5)

# Example usage of the trained model
question = 'What is the tallest mountain in the world?'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=64)
question_emb = question_encoder(question_tok)[0]
print(question_tok)
print(question_emb[:5])
