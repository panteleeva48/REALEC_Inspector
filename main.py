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
    result = {}
    result['av_depth'] = gf.av_depth()
    result['max_depth'] = gf.max_depth()
    result['min_depth'] = gf.min_depth()
    result['num_acl'], result['num_rel_cl'], result['num_advcl'] = gf.count_dep_sent()
    result['num_sent'] = gf.count_sent()
    result['num_tok'] = gf.count_tokens()
    result['av_tok_before_root'] = gf.tokens_before_root()
    result['av_len_sent'] = gf.mean_len_sent()
    result['num_cl'], result['num_tu'], result['num_compl_tu'] = gf.count_units()
    result['num_coord'] = gf.count_coord()
    result['num_poss'], result['num_prep'] = gf.count_poss_prep()
    result['num_adj_noun'] = gf.count_adj_noun()
    result['num_part_noun'] = gf.count_part_noun()
    result['num_noun_inf'] = gf.count_noun_inf()
    # result['pos_sim_nei'] = gf.pos_sim_mean2()
    # result['lemma_sim_nei'] = gf.lemma_sim_mean2()
    # result['pos_sim_all'] = gf.pos_sim_mean()
    # result['lemma_sim_all'] = gf.lemma_sim_mean()
    result['density'] = gf.density()
    result['ls'] = gf.LS()
    result['vs'], result['corrected_vs'], result['squared_vs'] =  gf.VS()
    result['lfp_1000'], result['lfp_2000'], result['lfp_uwl'], result['lfp_rest'] = gf.LFP()
    result['ndw'] = gf.NDW()
    result['ttr'], result['corrected_ttr'], result['root_ttr'], result['log_ttr'], result['uber_ttr'] = gf.TTR()
    result['d'] = gf.D()
    result['lv'] = gf.LV()
    result['vvi'], result['squared_vv'], result['corrected_vv'], result['vvii'] = gf.VV()
    result['nv'] = gf.NV()
    result['adjv'] = gf.AdjV()
    result['advv'] = gf.AdvV()
    result['modv'] = gf.ModV()
    result['der_level3'], result['der_level4'], result['der_level5'], result['der_level6'] = gf.derivational_suffixation()
    result['mci'] = gf.MCI()
    result['freq_finite_forms'] = gf.freq_finite_forms()
    result['freq_aux'] = gf.freq_aux()
    result['num_inf'], result['num_gerunds'], result['num_pres_sing'], result['num_pres_plur'], result['num_past_part'], result['num_past_simple'] = gf.num_verb_forms()
    result['num_linkings'] = gf.num_linkings().get('link_all')
    result['num_4grams'] = gf.num_4grams()
    result['num_func_ngrams'] = gf.num_func_ngrams().get('4grams_all')
    result['shell_noun'] = None
    return result


if __name__ == '__main__':
    PATH_TXT = os.path.join(BASE_DIR, 'data', 'test.txt')
    text = read_file(PATH_TXT)
    result = main(text)
    print(result)
