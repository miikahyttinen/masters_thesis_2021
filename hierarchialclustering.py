from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def ward_hierarchical_clustering(feature_matrix):
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix


def plot_hierarchical_clusters(linkage_matrix, titles, figure_size=(8, 12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    # TODO: move to pipeline
    ax = dendrogram(linkage_matrix, orientation="left", labels=titles)
    plt.tick_params(axis='x',
                    which='both', bottom='off', top='off', labelbottom='off')
    plt.tight_layout()
    plt.show()
