#!/usr/bin/env python3

import featureextractor
import kmeansclustering
import dbscan
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import operator

# <----- USAGE ----->
#
# NLP Command line tool for extracting clusters found with K-means and/or DBSCAN
#
#  - Can be used iteratively in hierarchical clustering style
#  - You will need a CSV-file including columns: name;category;data
#  - Columns "name" and "category" are just for tracking which sample is which.
#  - Column "data" needs to contain pre-processed natural language: no linebreaks, only space between words, utf-8
#  - File preprcessor.py can be used to pre-proces Finnish langauge. Google "NLTK" for English pre-processor
#  - Command "stop" will (over)write "clustered.csv" file with clusters column running label from 0 to n, zero meaning "no cluster"
#  - The tool will ask you to estimate some parameters like Epsilon for DBSCAN
#  - For K-means, the tool will automatically estimate number of clusters based on silhouette score value
#
# Author: miikahyttinen
# Licesne: GNU
#
# Provided "as is"
#
# In the end of this you will find an example of the usage.

# You can try different values.
MAX_NUM_OF_CLUSTERS = 50


def max_cluster_number_km(data_frame):
    sample_count = len(data_frame['data'].values.astype('U').tolist())
    if sample_count < MAX_NUM_OF_CLUSTERS:
        return sample_count
    else:
        return MAX_NUM_OF_CLUSTERS


def check_stop_command(command, data_frame_labelled):
    if command == 'stop':
        data_frame_labelled.to_csv(
            'clustered.csv', sep=';', encoding='utf-8', index=False)
        exit()


def get_drop_samples_index(uniq_chosen_labels, data_frame, cluster_labels, running_label):
    clustered_samples = []
    drop_index = []
    for uniq_label in uniq_chosen_labels:
        for index, l in enumerate(cluster_labels):
            if l == uniq_label:
                clustered_samples.append(
                    (list(data_frame['row_num'])[index], running_label))
                drop_index.append(index)
        running_label = running_label + 1
    return clustered_samples, drop_index, running_label


def nlp_pipeline(data_frame, data_frame_labelled, running_label):

    print('SHAPE OF THE UNCLUSTERED DATA FRAME: ' + str(data_frame.shape))

    try:
        N_SILHOUETTE_SCORE = max_cluster_number_km(data_frame)

        # Form TF-IDF feature matrix
        feature_names, feature_matrix = featureextractor.extract_features(
            data_frame['data'].values.astype('U').tolist())

        # Reduce to matrix to N components with SVD
        svd_feature_matrix = TruncatedSVD(
            n_components=5).fit_transform(feature_matrix)

        row_names = list(
            zip(data_frame['name'].values.tolist(), data_frame['category'].values.tolist()))
        km, clusters, scores = kmeansclustering.k_means_print_silhouette_scores(
            feature_matrix=svd_feature_matrix, num_clusters=N_SILHOUETTE_SCORE)

        # NOTE: Find the index of the highest silhouette score
        score_index, value = max(enumerate(scores), key=operator.itemgetter(1))

        # <------- K-MEANS ------>
        km_labels = kmeansclustering.k_means_svd_plot(
            svd_feature_matrix, row_names, score_index + 2)
        kmeansclustering.print_kmeans_clusters(km_labels, row_names)

        user_command = input('Use K means: y | n | stop ')
        check_stop_command(user_command, data_frame_labelled)

        # Choose clusters
        if user_command == 'y':

            unique_kmeans_labels = set(km_labels)

            print('K means cluster labels to choose from: ' +
                  str(unique_kmeans_labels))

            # Choose clusters to extract
            input_kmeans_labels = input('Choose labels: ')
            uniq_chosen_labels_km = [int(s)
                                     for s in input_kmeans_labels.split()]

            clustered_samples_km, drop_index_km, running_label = get_drop_samples_index(
                uniq_chosen_labels_km, data_frame, km_labels, running_label)

            # Drop clustered samples
            data_frame.drop(data_frame.index[drop_index_km], inplace=True)

            # Mark clusters
            for index, cluster_label in clustered_samples_km:
                data_frame_labelled.loc[index, 'cluster'] = cluster_label

            # Run clusterer again with un clustered data
            nlp_pipeline(data_frame, data_frame_labelled, running_label)

        # <------ DBSCAN ------->
        dbscan.find_epsilon(svd_feature_matrix)
        epsilon = input("Enter epsilon:")

        dbscan_labels = dbscan.dbscan(
            svd_feature_matrix, float(epsilon), 5, row_names)
        user_command = input('Use DBSCAN: y | n | stop ')

        if user_command == 'y':

            unique_dbscan_labels = set(dbscan_labels)

            print('DBSCAN cluster labels to choose from: ' +
                  str(unique_dbscan_labels))

            # Choose clusters to extract
            input_dbscan_labels = input('Choose labels: ')
            uniq_chosen_labels_dbscan = [int(s)
                                         for s in input_dbscan_labels.split()]

            clustered_samples_dbscan, drop_index_dbscan, running_label = get_drop_samples_index(
                uniq_chosen_labels_dbscan, data_frame, dbscan_labels, running_label)

            # Drop clustered samples
            data_frame.drop(data_frame.index[drop_index_dbscan], inplace=True)

            # Mark clusters chosen
            for index, cluster_label in clustered_samples_dbscan:
                data_frame_labelled.loc[index, 'cluster'] = cluster_label
            nlp_pipeline(data_frame, data_frame_labelled, running_label)

        check_stop_command(user_command, data_frame_labelled)

        nlp_pipeline(data_frame, data_frame_labelled, running_label)

    except Exception as e:
        print("Some error happened, starting again from where you left...")
        print("Error -- " + str(e))
        nlp_pipeline(data_frame, data_frame_labelled, running_label)


# <---- EXAMPLE ---->

csv_file_name = 'data_set_large_fin_pre_processed.csv'

data_frame_pre_processed = pd.read_csv(csv_file_name, sep=';')
data_frame_pre_processed['row_num'] = list(
    range(len(data_frame_pre_processed)))

data_frame_labelled = pd.read_csv(csv_file_name, sep=';')
data_frame_labelled['cluster'] = [0] * len(data_frame_pre_processed)
data_frame_labelled = data_frame_labelled.drop('data', 1)

nlp_pipeline(data_frame_pre_processed, data_frame_labelled, 1)
