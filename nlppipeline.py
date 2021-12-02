#!/usr/bin/env python3

from collections import Counter
from gensim import corpora, models
import preprocessor
import featureextractor
import kmeansclustering
import hierarchialclustering
import dbscan
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


def nlp_pipeline(data_frame, POS_CODES, SPLIT_COMPOUNDS, ITERATIONS=1):

    # PARAMETERS
    NUM_CLUSTERS = 200

    for it in range(2, 50, 5):
        print("DIM: " + str(it))
        all_scores = []
        # print('it: ' + str(it))
        preprocessed_data = preprocessor.pre_process(
            data_frame, POS_CODES, SPLIT_COMPOUNDS)
        feature_names, feature_matrix = featureextractor.extract_features(
            preprocessed_data)

        svd = TruncatedSVD(n_components=10).fit_transform(feature_matrix)

        # for col in feature_matrix:
        #    word_freq = np.count_nonzero(
        #        feature_matrix[col]) / len(feature_matrix[col])
        #    if word_freq > (0.5):
        #        # print('drop', col)
        #        feature_matrix.drop([col], axis=1)
        #        feature_names.remove(col)

        all_scores.append(scores)

        # data_frame['Cluster'] = clusters

        row_names = list(
            zip(data_frame['name'].values.tolist(), data_frame['category'].values.tolist()))

        # DBSCAN
        # dbscan.find_epsilon(feature_matrix)
        #dbscan.dbscan(feature_matrix, row_names)

        avarage_scores_for_k_clusters = [
            (sum(x) / ITERATIONS) for x in zip(*all_scores)]
        for s in avarage_scores_for_k_clusters:
            print(str(s).replace('.', ','))

    # print(len(summed_scores))

    # print("FREQ_CUTOFF:" + str(FREQ_CUTOFF))
    # print("AVG: " + str(np.mean(avarage_scores_for_k_clusters)))
    # print("MAX: " + str(max(avarage_scores_for_k_clusters)))
    # print("MIN: " + str(min(avarage_scores_for_k_clusters)))
    # linkage_matrix = hierarchialclustering.ward_hierarchical_clustering(
    #    feature_matrix)
    # hierarchialclustering.plot_hierarchical_clusters(
    #    linkage_matrix, data_frame, titles, figure_size=(8, 10))

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

COMBINED_DATA_SET = DATA_SET_TRADE_LONG[0].append(
    DATA_SET_IT_SHORT[0], ignore_index=True)

data_sets = [DATA_SET_TRADE_SHORT, DATA_SET_TRADE_LONG,
             DATA_SET_IT_SHORT, DATA_SET_IT_LONG]

POS_CODES = [NOUN]
POS_ALL = ['all']

nlp_pipeline(DATA_SET_LARGE[0], POS_CODES, True,
             1, 'large_pre_processed_fin.txt')

# print('#### ALL, NO COMPUND SPLITTING')

# for data in data_sets:
#    print('------------' + data[1] + '-------------')
#    nlp_pipeline(data[0], POS_ALL, False, 10)

# print('#### NOUNS, NO COMPUND SPLITTING ####')

# for data in data_sets:
#    print('------------' + data[1] + '-------------')
#    nlp_pipeline(data[0], POS_CODES, False, 10)

# print('#### NOUNS, WITH COMPUND SPLITTING ####')

# for data in data_sets:
#    print('------------' + data[1] + '-------------')
