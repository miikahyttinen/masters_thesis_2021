from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def display_features(features, feature_names):
    df = pd.DataFrame(data=features, columns=feature_names)
    return df


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def extract_features(preprocessed_data):
    bow_vectorizer, bow_features = bow_extractor(preprocessed_data)
    tfidf_trans, transformed_features = tfidf_transformer(bow_features)
    feature_names = bow_vectorizer.get_feature_names()
    features = np.round(transformed_features.todense(), 2)
    return feature_names, display_features(features, feature_names)
