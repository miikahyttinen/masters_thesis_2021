#!/usr/bin/env python3

from urllib.parse import unquote
import libvoikko
import re

v = libvoikko.Voikko(u'fi')


def clean_corpus(corpus, unwanted='.,!?"><'):
    # replace %xx escapes (a.k.a ASCII encodings) and lowercase corpus
    encoded = unquote(corpus).lower()
    for c in unwanted:
        encoded.replace(c, '')
    return encoded.replace('-', '')

# Halutaanko jättää yhdyssanat


def reduce_compounds(word, pos_codes):
    bases = v.analyze(word)
    if len(bases) == 0:
        return ''
    # find wordbases which Voikko places inside of parenthesis
    result = re.findall('\(.*?\)', bases[0]['WORDBASES'])
    splitted = ''
    for r in result:
        # parse single words from Voikko's compound represantation
        if not '+' in r:
            cleaned = r.replace('(', '').replace(')', '').replace('=', '')
            voikko_dict = v.analyze(cleaned)
            split = ''
            if voikko_dict:
                split = voikko_dict[0]['BASEFORM'] if voikko_dict[0]['CLASS'] in pos_codes else ''
            splitted = splitted + ' ' + split
    return splitted


def voikko_analysis(w, pos_codes, split_compounds):
    # POS-tagging, word recognizing, Finnish language lemmatization
    voikko_dict = v.analyze(w)
    if not voikko_dict:
        return ''
    if 'all' in pos_codes:
        return voikko_dict[0]['BASEFORM']
    else:
        s = ''
        voikko_baseform = voikko_dict[0]['BASEFORM'] if voikko_dict[0]['CLASS'] in pos_codes else ''
        if not(voikko_baseform == ''):
            if split_compounds:
                return(reduce_compounds(voikko_baseform, pos_codes))
            else:
                return voikko_baseform
        else:
            return ''


def filter_whitespace(s):
    if s == '':
        return False
    return True


def lemmatize(words, pos_codes, split_compounds):
    words = (list(map(lambda word: voikko_analysis(
        word, pos_codes, split_compounds), words)))
    words_iterator = filter(filter_whitespace, words)
    return ' '.join(list(words_iterator))


def tokenize_corpora(corpora):
    return list(map(lambda corpus: corpus.split(' '), corpora))


def pre_process(data_frame, pos_codes, split_compounds=False):

    corpus_column = data_frame['data'].tolist()

    cleaned_corpora = list(
        map(lambda row: clean_corpus(row), corpus_column))
    # Tsekkaa NLTK stop words´

    tokenized_data = tokenize_corpora(cleaned_corpora)

    lemmatized_data = list(
        map(lambda tokenized: lemmatize(tokenized, pos_codes, split_compounds), tokenized_data))

    return lemmatized_data
