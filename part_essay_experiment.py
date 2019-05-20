import os
from utils.get_feature_values import GetFeatures
from config import UDPIPE_MODEL, BASE_DIR, chkr, BASE_DIR
import pickle
from get_db import result
import pandas as pd
import json
import nltk


DATASET = os.path.join(BASE_DIR, 'data', 'part_experiment_result.csv')
JSON_FILE = os.path.join(BASE_DIR, 'data', 'files_with_json.txt')
RESULT_FILE = os.path.join(BASE_DIR, 'data', 'result.json')
gf = GetFeatures(UDPIPE_MODEL)


def tokenizer(text):
    return text.split()


DON_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'shell.pickle')
with open(DON_MODEL_PATH, 'rb') as mdl:
    DON_MODEL = pickle.load(mdl)


def check_spelling(text):
    chkr.set_text(text)
    for err in chkr:
        suggestions = err.suggest()
        if suggestions:
            suggest = suggestions[0]
            err.replace(suggest)
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
    result['pos_sim_nei'], result['lemma_sim_nei'] = gf.simularity_neibour()
    result['pos_sim_all'], result['lemma_sim_all'] = gf.simularity_mean()
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
    (result['der_level3'], result['der_level4'],
     result['der_level5'], result['der_level6']) = gf.derivational_suffixation()
    result['mci'] = gf.MCI()
    result['freq_finite_forms'] = gf.freq_finite_forms()
    result['freq_aux'] = gf.freq_aux()
    (result['num_inf'], result['num_gerunds'], result['num_pres_sing'],
     result['num_pres_plur'], result['num_past_part'], result['num_past_simple']) = gf.num_verb_forms()
    result['num_linkings'] = gf.num_linkings().get('link_all')
    result['num_4grams'] = gf.num_4grams()
    result['num_func_ngrams'] = gf.num_func_ngrams().get('4grams_all')
    result['num_shell_noun'] = gf.shell_nouns(DON_MODEL)
    result['num_misspelled_tokens'] = gf.number_of_misspelled()
    result['punct_mistakes_pp'] = gf.count_punct_mistakes_participle_phrase()
    result['punct_mistakes_because'] = gf.count_punct_mistakes_before(before='because')
    result['punct_mistakes_but'] = gf.count_punct_mistakes_before(before='but')
    result['punct_mistakes_compare'] = gf.count_punct_mistakes_before(before='than') \
                                       + gf.count_punct_mistakes_before(before='like')
    result['million_mistake'] = gf.count_million_mistakes()
    result['side_mistake'] = gf.if_side_mistake()
    return result


if __name__ == '__main__':
    with open(RESULT_FILE) as data_file:
        data = json.load(data_file)
    with open(JSON_FILE, 'r') as rf:
        clean_essays = rf.read().split('\n')
    data['name'] = []
    data['text'] = []
    data['class'] = []
    data['type'] = []
    data['part'] = []
    for i, essay in enumerate(result):
        if i % 10 == 0:
            print(i + 1, 'files are parsed.')
        text = essay[0]
        mark = essay[1]
        name = essay[2]
        type = essay[3]
        if name in clean_essays:
            sentences = nltk.sent_tokenize(text)
            half = int(len(sentences) / 2) + 1
            first_part = ' '.join(sentences[:half])
            second_part = ' '.join(sentences[half:])
            try:
                for p, part in enumerate([first_part, second_part]):
                    _result = main(part)
                    for key in _result:
                        data[key].append(_result[key])
                    data['name'].append(name)
                    data['text'].append(part)
                    data['class'].append(mark)
                    data['type'].append(type)
                    data['part'].append(p)
            except:
                print('problem', name)
                continue

    df = pd.DataFrame(data=data)
    df.to_csv(DATASET, index=False)
