import hdbscan
from numpy.core.records import array
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

viridis = cm.get_cmap('viridis', 8)


def plot_hdbscan_cluster(X, row_names):
    clusterer = hdbscan.HDBSCAN(metric='euclidean')
    clusterer.fit(X)
    plot_svd = TruncatedSVD(n_components=2).fit_transform(X)
    plot_and_labels = zip(plot_svd, clusterer.labels_, row_names)
    print(plot_and_labels)
    for plot, label, name in plot_and_labels:
        x = plot[0]
        y = plot[1]
        if not label == -1:
            color = cm.hot(label)
            plt.plot(x, y, color=viridis(label), marker='o')
            # plt.annotate(name[1], (x, y))
        else:
            plt.plot(x, y, color='black', marker='x',)
    # fig, ax = plt.subplots(figsize=(16, 8))
    plt.show()
    # ax.scatter(plot_svd[:, 0], plot_svd[:, 1], s=50,
    #           linewidths=0.5, alpha=0.7, c=(colored_labels))
    # for i, txt in enumerate(row_names):
    #    ax.annotate(txt[1] + txt[0], (plot_svd[i, 0], plot_svd[i, 1]))


def print_hdbscan_cluster(X, row_names):
    clusterer = hdbscan.HDBSCAN(metric='euclidean')
    clusterer.fit(X)
    labels_and_names = zip(clusterer.labels_, row_names)
    sorted_by_cluster = sorted(labels_and_names, key=lambda tup: tup[0])
    for s in sorted_by_cluster:
        print('Cluster: ' + str(s[0]) + ' Info: ' + str(s[1]))
