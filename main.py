import os
from utils.get_feature_values import GetFeatures
from config import UDPIPE_MODEL, BASE_DIR, chkr
gf = GetFeatures(UDPIPE_MODEL)


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
    num_cl, num_tu, num_compl_tu = gf.count_units()


if __name__ == '__main__':
    #PATH_TXT = os.path.join(BASE_DIR, 'data', 'test.txt')
    PATH_TXT = '/Users/ira/Downloads/TUnit.txt'
    text = read_file(PATH_TXT)
    result = main(text)
    print(result)
