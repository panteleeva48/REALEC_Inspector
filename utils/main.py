import os

from utils.get_feature_values import GetFeatures
from config import UDPIPE_MODEL, BASE_DIR

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
    result = {}
    result['density'] = gf.density()
    result['LS'] = gf.LS()
    result['VSI'], result['VSII'], result['VSIII'] = gf.VS()
    result['LFP_first_procent'], result['LFP_second_procent'],\
    result['LFP_third_procent'], result['LFP_none'] = gf.LFP()
    result['NDW'] = gf.NDW()
    result['TTR'], result['CTTR'], result['RTTR'], result['LogTTR'], result['Uber'] = gf.TTR()
    result['D'] = gf.D()
    result['LV'] = gf.LV()
    result['VVI'], result['SVVI'], result['CVVI'], result['VVII'] = gf.VV()
    result['NV'] = gf.NV()
    result['AdjV'] = gf.AdjV()
    result['AdvV'] = gf.AdvV()
    result['ModV'] = gf.ModV()
    result['der_suff3'], result['der_suff4'], result['der_suff5'], result['der_suff6'] = gf.derivational_suffixation()
    result['MCI'] = gf.MCI()
    result['freq_finite_forms'] = gf.freq_finite_forms()
    result['freq_aux'] = gf.freq_aux()
    result['infinitive_tokens'], result['gerund_tokens'],\
    result['pres_sg_tokens'], result['pres_pl_tokens'], result['parts'], result['pasts'] = gf.num_verb_forms()
    links = gf.num_linkings()
    for link in links:
        result[link] = links[link]
    result['num_4grams'] = gf.num_4grams()
    result['num_func_ngrams'] = gf.num_func_ngrams()
    result['av_depth'] = gf.av_depth()
    result['max_depth'] = gf.max_depth()
    result['min_depth'] = gf.min_depth()
    result['acl'], result['rel_cl'], result['advcl'] = gf.count_dep_sent()
    result['count_sent'] = gf.count_sent()
    result['count_tokens'] = gf.count_tokens()
    result['tokens_before_root'] = gf.tokens_before_root()
    result['mean_len_sent'] = gf.mean_len_sent()
    return result


if __name__ == '__main__':
    PATH_TXT = os.path.join(BASE_DIR, 'data', 'test.txt')
    result = main(PATH_TXT)
    print(result)
