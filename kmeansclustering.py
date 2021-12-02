from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import random
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD


def k_means_svd_plot(svd_feature_matrix, row_names, num_clusters):
    plot_svd = TruncatedSVD(n_components=2).fit_transform(svd_feature_matrix)
    km_obj, clusters = k_means(
        feature_matrix=svd_feature_matrix, num_clusters=num_clusters)
    fig, ax = plt.subplots(figsize=(16, 8))
    print(km_obj.labels_ + 1)

    unique_labels = set(km_obj.labels_)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    ax.scatter(plot_svd[:, 0], plot_svd[:, 1], s=50,
               linewidths=0.5, alpha=0.7, c=(km_obj.labels_ * 2), cmap='rainbow')
    # for i, txt in enumerate(row_names):
    #    ax.annotate(txt[1], (plot_svd[i, 0], plot_svd[i, 1]))
    print_kmeans_clusters(km_obj.labels_, row_names)
    plt.show()
    return km_obj.labels_


def print_kmeans_clusters(labels, row_names):
    labels_and_names = zip(labels, row_names)
    sorted_by_cluster = sorted(labels_and_names, key=lambda tup: tup[0])
    for s in sorted_by_cluster:
        print('Cluster: ' + str(s[0]) + ' Info: ' + str(s[1]))


def k_means(feature_matrix, num_clusters):
    km = KMeans(n_clusters=num_clusters, max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


def k_means_print_silhouette_scores(feature_matrix, num_clusters):
    scores = []
    for n in range(2, num_clusters + 1):
        km = KMeans(n_clusters=n, max_iter=10000)
        km.fit(feature_matrix)
        score = silhouette_score(feature_matrix, km.labels_)
        print('n' + ';' + str(n) + ';' + 'silhouette_score' + ';' + str(score))
        scores.append(score)
        clusters = km.labels_
    return km, clusters, scores


def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-'*20)
        print('Key features:', cluster_details['key_features'])
        print('Companies in this cluster:')
        print(', '.join(cluster_details['companies']))
        print('='*40)


def get_cluster_data_svd(clustering_obj, data_frame, num_clusters):
    cluster_details = {}
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        companies = data_frame[data_frame['Cluster']
                               == cluster_num]['name'].values.tolist()
        cluster_details[cluster_num]['companies'] = companies
    return cluster_details


def get_cluster_data(clustering_obj, data_frame, feature_names, num_clusters, topn_features=10):
    cluster_details = {}
    # get cluster centroids
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # get key features for each cluster
    # get movies belonging to each cluster
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index]
                        for index in ordered_centroids[cluster_num, :topn_features]]

        cluster_details[cluster_num]['key_features'] = key_features

        companies = data_frame[data_frame['Cluster']
                               == cluster_num]['name'].values.tolist()
        cluster_details[cluster_num]['companies'] = companies
    return cluster_details


def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, data_frame, titles, plot_size=(16, 8)):
    # generate random color for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color
    # define markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*',
               'h', 'H', 'D', 'd']  # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data.items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(
            cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and movies
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': data_frame['Cluster'].values.tolist(),
                                       'title': titles
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each cluster using co-ordinates and movie titles
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'], marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num], color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off',
                       top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off',
                       top='off', labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),
                  fancybox=True, shadow=True, ncol=5, numpoints=1, prop=fontP)
    # add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.loc[index]['x'], cluster_plot_frame.loc[index]
                ['y'], cluster_plot_frame.loc[index]['title'], size=8)
    # show the plot
    plt.show()
