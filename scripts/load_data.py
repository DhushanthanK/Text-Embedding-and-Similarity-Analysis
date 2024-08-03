import gensim.downloader as api
import pandas as pd

# Load the pre-trained word vectors (GloVe model)
word_vectors = api.load('glove-wiki-gigaword-100')

# Example function to convert words to vectors
def word_to_vector(text):
    # Tokenize the text into words
    words = text.split()  # Simple tokenization by splitting on spaces
    vectors = []
    for word in words:
        try:
            vectors.append(word_vectors[word])
        except KeyError:
            print(f"Word '{word}' not found in the model.")
            vectors.append(None)  # Handle missing words
    return vectors

# Load your data
data_path = 'data/nq_sample.tsv'
data = pd.read_csv(data_path, sep='\t')

# Print the column names to verify
print("Columns in the DataFrame:", data.columns)

# Use the correct column names from the DataFrame
if 'question' in data.columns:
    data['question_vectors'] = data['question'].apply(lambda x: word_to_vector(x) if pd.notna(x) else None)
else:
    print("Column 'question' does not exist in the DataFrame.")

if 'answer' in data.columns:
    data['answer_vectors'] = data['answer'].apply(lambda x: word_to_vector(x) if pd.notna(x) else None)
else:
    print("Column 'answer' does not exist in the DataFrame.")

# Example: Print the first few rows of the dataframe
print(data.head())
