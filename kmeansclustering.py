import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD


def k_means_svd_plot(svd_feature_matrix, row_names, num_clusters):
    plot_svd = TruncatedSVD(n_components=2).fit_transform(svd_feature_matrix)
    km_obj, clusters = k_means(
        feature_matrix=svd_feature_matrix, num_clusters=num_clusters)
    fig, ax = plt.subplots(figsize=(16, 8))

    unique_labels = set(km_obj.labels_)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    ax.scatter(plot_svd[:, 0], plot_svd[:, 1], s=50,
               linewidths=0.5, alpha=0.7, c=(km_obj.labels_ * 2), cmap='rainbow')
    # Remvoe comment if you want to plot sample names and categories. Not recommended with large data sets!
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
        print('n' + '=' + str(n) + ' ' + ' Silhouette score ' + '=' + str(score))
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
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
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
