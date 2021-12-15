#!/usr/bin/env python3

import preprocessor
import pandas as pd

NOUN = 'nimisana'
VERB = 'teonsana'
ADJECTIVE = 'laatusana'

SPLIT_COMPUNDS = True
POS_ALL = ['all']

POS_CODES = [NOUN]

data_frame = pd.read_csv('data_set_large_short_fin.csv', sep=';')

# SAVE PRE PROCESSED DATA AS CSV
preprocessed_data = preprocessor.pre_process(
    data_frame, POS_CODES, SPLIT_COMPUNDS)
data_frame['data'] = preprocessed_data
data_frame.to_csv('data_set_large_fin_pre_processed.csv',
                  sep=';', encoding='utf-8', index=False)
