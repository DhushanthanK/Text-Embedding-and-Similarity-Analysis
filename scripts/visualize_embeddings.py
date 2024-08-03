import sys
import os
import matplotlib.pyplot as plt


# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from visualize import plot_similarity

import numpy as np

messages = [
    "I like my phone",
    "My phone is not good.",
    "Your cellphone looks great.",
    "Will it snow tomorrow?",
    "Recently a lot of hurricanes have hit the US",
    "Global warming is real",
    "An apple a day, keeps the doctors away",
    "Eating strawberries is healthy",
    "Is paleo better than keto?",
    "How old are you?",
    "what is your age?",
]

# Assume `embeddings` are already computed
embeddings = np.random.rand(len(messages), 768)  # Placeholder for actual embeddings
plot_similarity(messages, embeddings, 90)
plt.show()
