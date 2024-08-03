import matplotlib.pyplot as plt
import seaborn as sns

from embedding_utils import cosine_similarity_matrix

from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_matrix(features):
    return cosine_similarity(features)

def plot_similarity(labels, features, rotation):
    sim = cosine_similarity_matrix(features)
    sns.set_theme(font_scale=1.2)
    g = sns.heatmap(sim, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    return g
