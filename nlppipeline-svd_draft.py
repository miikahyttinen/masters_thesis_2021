#!/usr/bin/env python3

from collections import Counter
from gensim import corpora, models
import preprocessor
import featureextractor
import kmeansclustering
import dbscan
import hierarchialclustering
import hdbscan_clustering
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib.markers as CARETDOWNBASE


def nlp_pipeline(data_frame, POS_CODES, SPLIT_COMPOUNDS, PRE_PROCESSED_FILE=''):

    # PARAMETERS
    NUM_CLUSTERS = 2

    if PRE_PROCESSED_FILE == '':
        preprocessed_data = preprocessor.pre_process(
            data_frame, POS_CODES, SPLIT_COMPOUNDS)
    else:
        with open(PRE_PROCESSED_FILE) as file:
            preprocessed_data = [line.strip() for line in file]

    feature_names, feature_matrix = featureextractor.extract_features(
        preprocessed_data)

    print('SHAPE OF THE FEATURE MATRIX:' + str(feature_matrix.shape))

    svd_feature_matrix = TruncatedSVD(
        n_components=8).fit_transform(feature_matrix)

    row_names = list(
        zip(data_frame['name'].values.tolist(), data_frame['category'].values.tolist()))

    # SVD INSPECT VARIANCE
    # svd = TruncatedSVD(n_components=100).fit(feature_matrix)
    # for v in svd.explained_variance_ratio_:
    #    print(v)

    # K-MEANS
    # km_obj, clusters, scores = kmeansclustering.k_means(
    #    feature_matrix=feature_matrix, num_clusters=NUM_CLUSTERS)
    # tuple[0] + ' ' + tuple[1]
    #titles = list(map(lambda tuple: '', row_names))
    #data_frame['Cluster'] = clusters
    # cluster_data = kmeansclustering.get_cluster_data(
    #    km_obj, data_frame, feature_names, NUM_CLUSTERS, topn_features=10)
    # kmeansclustering.plot_clusters(
    #    NUM_CLUSTERS, feature_matrix, cluster_data, data_frame, titles, plot_size=(16, 8))

    # K-MEANS SVD
    # kmeansclustering.k_means_svd_plot(
    #    svd_feature_matrix, row_names, NUM_CLUSTERS)

    # SILHOUETTE SCORES
    # km_obj, clusters, scores = kmeansclustering.k_means_print_silhouette_scores(
    #    feature_matrix=feature_matrix, num_clusters=NUM_CLUSTERS)

    # WARD'S HIERARCHIAL CLUSTERING
    # linkage_matrix = hierarchialclustering.ward_hierarchical_clustering(
    #    svd_feature_matrix)
    # hierarchialclustering.plot_hierarchical_clusters(
    #    linkage_matrix, row_names, figure_size=(8, 10))

    # DBSCAN
    # dbscan.find_epsilon(svd_feature_matrix)
    dbscan.dbscan(svd_feature_matrix, 0.15, 10, row_names)
    # dbscan.print_dbscan_clusters(svd_feature_matrix, row_names, 0.15, 10)

    # HBDSCAN
    # hdbscan_clustering.print_hdbscan_cluster(svd_feature_matrix, row_names)

    # PRINT SCORES FOR DIFFERENT N:s
    # avarage_scores_for_k_clusters = [
    #    (sum(x) / ITERATIONS) for x in zip(*all_scores)]
    # for s in avarage_scores_for_k_clusters:
    #    print(str(s).replace('.', ','))

    # PRINT SCORES FOR DIFFERENT NUMBER OF COMPONENTS
    # i = 2
    # for score in all_scores:
    #    print(str(i) + ': ' + str(score))
    #    i = i + 1


 # VOIKKO POS CODES
NOUN = 'nimisana'
VERB = 'teonsana'
ADJECTIVE = 'laatusana'

DATA_SET_LARGE = (pd.read_csv(
    'data_set_large_short_fin.csv', sep=';'), 'TRADE SHORT')
DATA_SET_TRADE_SHORT = (pd.read_csv(
    'data_set_trade_short_fin.csv', sep=';'), 'TRADE SHORT')
DATA_SET_TRADE_LONG = (pd.read_csv(
    'data_set_trade_long_fin.csv', sep=';'), 'TRADE LONG')
DATA_SET_IT_SHORT = (pd.read_csv(
    'data_set_it_short_fin.csv', sep=';'), 'IT SHORT')
DATA_SET_IT_LONG = (pd.read_csv(
    'data_set_it_long_fin.csv', sep=';'), 'IT LONG')

COMBINED_DATA_SET_LONG = DATA_SET_TRADE_LONG[0].append(
    DATA_SET_IT_LONG[0], ignore_index=True)

COMBINED_DATA_SET_SHORT = DATA_SET_TRADE_SHORT[0].append(
    DATA_SET_IT_LONG[0], ignore_index=True)

POS_CODES = [NOUN]
POS_ALL = ['all']

# LARGE DATA SET
nlp_pipeline(DATA_SET_LARGE[0], POS_CODES, True, 'large_pre_processed_fin.txt')

#nlp_pipeline(COMBINED_DATA_SET_SHORT, POS_CODES, True)


# print('#### ALL, NO COMPUND SPLITTING')
# print('### SHORT ###')
# nlp_pipeline(COMBINED_DATA_SET_SHORT, POS_ALL, False)
# print('### LONG ###')
# nlp_pipeline(COMBINED_DATA_SET_LONG, POS_ALL, False)
#
# print('#### NOUNS, NO COMPUND SPLITTING')
# print('### SHORT ###')
# nlp_pipeline(COMBINED_DATA_SET_SHORT, POS_CODES, False)
# print('### LONG ###')
# nlp_pipeline(COMBINED_DATA_SET_LONG, POS_CODES, False)
#
# print('#### NOUNS, WITH COMPOUND SPLITTING')
# print('### SHORT ###')
# print('### LONG ###')
# nlp_pipeline(COMBINED_DATA_SET_LONG, POS_CODES, True)
