#!/usr/bin/env python3

import preprocessor
import pandas as pd

NOUN = 'nimisana'
VERB = 'teonsana'
ADJECTIVE = 'laatusana'

POS_CODES = [NOUN]
POS_ALL = ['all']

data_frame = pd.read_csv('data_set_large_short_fin.csv', sep=';')

# SAVE PRE PROCESSED DATA AS CSV
preprocessed_data = preprocessor.pre_process(data_frame, [NOUN], True)
data_frame['data'] = preprocessed_data
data_frame.to_csv('data_set_large_fin_pre_processed.csv',
                  sep=';', encoding='utf-8', index=False)
#textfile = open('round_five_pre_processed_fin', "w")
# for s in preprocessed_data:
#    textfile.write(s + "\n")
# textfile.close()

# nlp_pipeline(data_frame, [NOUN], True, 'round_five_pre_processed_fin.txt')
