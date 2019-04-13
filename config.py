import os
from utils.model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_UDPIPE_MODEL = os.path.join(BASE_DIR, 'models/english-partut-ud-2.3-181115.udpipe')
UDPIPE_MODEL = Model(PATH_UDPIPE_MODEL)