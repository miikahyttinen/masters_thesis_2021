#!/usr/bin/env python3

from enum import unique
import preprocessor
import featureextractor
import kmeansclustering
import dbscan
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import operator


def nlp_pipeline(data_frame, data_frame_static, running_label):

    print(list(data_frame['row_num']))

    print('Running label ------> ' + str(running_label))
    print('SHAPE OF THE DATA FRAME:' + str(data_frame.shape))

# try:
    sample_count = len(data_frame['data'].values.astype('U').tolist())
    if sample_count < 40:
        N_SILHOUETTE_SCORE = sample_count
    else:
        N_SILHOUETTE_SCORE = 40
    # PARAMETERS
    feature_names, feature_matrix = featureextractor.extract_features(
        data_frame['data'].values.astype('U').tolist())
    svd_feature_matrix = TruncatedSVD(
        n_components=6).fit_transform(feature_matrix)
    row_names = list(
        zip(data_frame['name'].values.tolist(), data_frame['category'].values.tolist()))
    km, clusters, scores = kmeansclustering.k_means_print_silhouette_scores(
        feature_matrix=svd_feature_matrix, num_clusters=N_SILHOUETTE_SCORE)
    # NOTE: Number of clusters is index + 2
    score_index, value = max(enumerate(scores), key=operator.itemgetter(1))
    # K-MEANS SVD
    km_labels = kmeansclustering.k_means_svd_plot(
        svd_feature_matrix, row_names, score_index + 2)
    kmeansclustering.print_kmeans_clusters(km_labels, row_names)
    use_kmeans = input('Use K means: y | n | stop ')
    if use_kmeans == 'stop':
        data_frame_static.to_csv(
            'clustered.csv', sep=';', encoding='utf-8', index=False)
        exit()
    clustered_samples = []
    drop_index = []
    # Choose clusters
    if use_kmeans == 'y':
        unique_kmeans_labels = set(km_labels)
        print('K means cluster labels to choose from: ' +
              str(unique_kmeans_labels))
        input_kmeans_labels = input('Choose labels: ')
        uniq_chosen_labels = [int(s) for s in input_kmeans_labels.split()]
        for uniq_label in uniq_chosen_labels:
            for index, l in enumerate(km_labels):
                if l == uniq_label:
                    clustered_samples.append(
                        (list(data_frame['row_num'])[index], running_label))
                    drop_index.append(index)
            running_label = running_label + 1
        # Drop clustered samples
        data_frame.drop(data_frame.index[drop_index], inplace=True)
        # Mark clusters
        for index, cluster_label in clustered_samples:
            print('index:' + str(index))
            print('running: ' + str(cluster_label))
            data_frame_static.loc[index, 'cluster'] = cluster_label
        nlp_pipeline(data_frame, data_frame_static, running_label)

    # DBSCAN
    clustered_samples = []
    drop_index = []
    dbscan.find_epsilon(svd_feature_matrix)
    epsilon = input("Enter epsilon:")
    dbscan_labels = dbscan.dbscan(
        svd_feature_matrix, float(epsilon), 5, row_names)
    use_dbscan = input('Use DBSCAN: y | n | stop ')
    if use_dbscan == 'y':
        unique_dbscan_labels = set(dbscan_labels)
        print('DBSCAN cluster labels to choose from: ' +
              str(unique_dbscan_labels))
        # Choose clusters
        input_dbscan_labels = input('Choose labels: ')
        uniq_chosen_labels = [int(s) for s in input_dbscan_labels.split()]
        for uniq_label in uniq_chosen_labels:
            for index, l in enumerate(dbscan_labels):
                if l == uniq_label:
                    clustered_samples.append(
                        (list(data_frame['row_num'])[index], running_label))
                    drop_index.append(index)
            running_label = running_label + 1
        # Drop clustered samples
        data_frame.drop(data_frame.index[drop_index], inplace=True)
        # Mark clusters
        for index, cluster_label in clustered_samples:
            print('index:' + str(index))
            print('running: ' + str(cluster_label))
            data_frame_static.loc[index, 'cluster'] = cluster_label
        nlp_pipeline(data_frame, data_frame_static, running_label)
    if use_dbscan == 'stop':
        data_frame_static.to_csv(
            'clustered.csv', sep=';', encoding='utf-8', index=False)
        exit()
    nlp_pipeline(data_frame, data_frame_static, running_label)
# except Exception as e:
#    print("Some error happened, starting from where you left...")
#    print("Error -- " + str(e))
#    nlp_pipeline(data_frame, data_frame_static, running_label)


data_frame_pre_processed = pd.read_csv(
    'data_set_large_fin_pre_processed.csv', sep=';')
data_frame_pre_processed['row_num'] = list(
    range(len(data_frame_pre_processed)))

print(list(
    range(len(data_frame_pre_processed))))

data_frame_static = pd.read_csv(
    'data_set_large_fin_pre_processed.csv', sep=';')
data_frame_static['cluster'] = [0] * len(data_frame_pre_processed)
data_frame_static = data_frame_static.drop('data', 1)

nlp_pipeline(data_frame_pre_processed, data_frame_static, 1)
