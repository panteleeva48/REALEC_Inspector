import os
from utils.model import Model
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_UDPIPE_MODEL = os.path.join(BASE_DIR, 'models', 'english-partut-ud-2.3-181115.udpipe')
UDPIPE_MODEL = Model(PATH_UDPIPE_MODEL)

PATH_LISTS = os.path.join(BASE_DIR, 'data', 'lists', 'lists.json')
with open(PATH_LISTS) as data_file:
    lists = json.load(data_file)
FIVE_T_FREQ_COCA = lists['5000frequentCOCA']
FREQ_VERBS_COCA_FROM_FIVE_T = lists['frequentverbsCOCAfrom5000']
UWL = lists['UWL']

OPEN_CLASS = ['NOUN', 'VERB', 'ADV', 'ADJ']

PATH_LINKINGS = os.path.join(BASE_DIR, 'data', 'lists', 'linkings.json')
with open(PATH_LINKINGS) as data_file:
    LINKINGS = json.load(data_file)

PATH_FUNC_NGRAMS = os.path.join(BASE_DIR, 'data', 'lists', 'functional_ngrams.json')
with open(PATH_FUNC_NGRAMS) as data_file:
    FUNC_NGRAMS = json.load(data_file)

PATH_SUFFIXES = os.path.join(BASE_DIR, 'data', 'lists', 'suffixes.json')
with open(PATH_SUFFIXES) as data_file:
    SUFFIXES = json.load(data_file)

PATH_NGRAMS = os.path.join(BASE_DIR, 'data', 'lists', 'ngrams.txt')
with open(PATH_NGRAMS) as data_file:
    NGRAMS = [x.split() for x in data_file.read().split('\n')]