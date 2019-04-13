from get_feature_values import GetFeatures
from config import UDPIPE_MODEL, BASE_DIR
import os
import enchant.checker as spellcheck

chkr = spellcheck.SpellChecker("en_GB")
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


def main(path):
    text = read_file(path)
    text = check_spelling(text)
    gf.get_info(text)


if __name__ == '__main__':
    PATH_TXT = os.path.join(BASE_DIR, 'data', 'test.txt')
    main(PATH_TXT)
