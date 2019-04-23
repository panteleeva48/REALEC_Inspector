import os
from utils.get_feature_values import GetFeatures
from config import UDPIPE_MODEL, BASE_DIR, chkr
gf = GetFeatures(UDPIPE_MODEL)
# PROPN ? как что считать

def check_spelling(text):
    chkr.set_text(text)
    for err in chkr:
        sug = err.suggest()[0]
        err.replace(sug)
    text = chkr.get_text()
    return text


def read_file(path):
    with open(path, 'r', encoding='utf-8') as fr:
        text = fr.read()
    return text


def main(text):
    text = check_spelling(text)
    gf.get_info(text)
    print(text)
    adj_noun = gf.count_adj_noun()
    print(adj_noun)
    part_noun = gf.count_part_noun()
    print(part_noun)
    inf_noun = gf.count_noun_inf()
    print(inf_noun)


if __name__ == '__main__':
    # PATH_TXT = os.path.join(BASE_DIR, 'data', 'test.txt')
    PATH_TXT = '/Users/ira/Downloads/REALEC_Inspector/data/noun.txt'
    text = read_file(PATH_TXT)
    main(text)
