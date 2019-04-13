import collections
from config import FIVE_T_FREQ_COCA, FREQ_VERBS_COCA_FROM_FIVE_T, UWL, LINKINGS, FUNC_NGRAMS
from operations import safe_divide, division, corrected_division, root_division, squared_division, log_division, uber
import random
import re

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

from parser import ParserUDpipe
parser = ParserUDpipe()


class GetFeatures:
    """Returns values of complexity criteria."""

    def __init__(self, model):
        self.model = model
        self.text = ''
        self.lemmas = []
        self.tokens = []
        self.verb_lemmas = []
        self.noun_lemmas = []
        self.adj_lemmas = []
        self.adv_lemmas = []
        self.open_class_lemmas = []
        self.infinitive_tokens = []
        self.gerund_tokens = []
        self.pres_sg_tokens = []
        self.verb_tokens = []
        self.aux_forms = []
        self.pres_pl_tokens = []
        self.parts = []
        self.pasts = []
        self.finite_tokens = []

    def get_info(self, text):
        self.text = text
        parser.text2conllu(self.text, self.model)
        parser.get_info()
        self.lemmas = parser.lemmas
        self.tokens = parser.tokens
        self.verb_lemmas = parser.verb_lemmas
        self.noun_lemmas = parser.noun_lemmas
        self.adj_lemmas = parser.adj_lemmas
        self.adv_lemmas = parser.adv_lemmas
        self.open_class_lemmas = parser.open_class_lemmas
        self.infinitive_tokens = parser.infinitive_tokens
        self.gerund_tokens = parser.gerund_tokens
        self.pres_sg_tokens = parser.pres_sg_tokens
        self.verb_tokens = parser.verb_tokens
        self.aux_forms = parser.aux_forms
        self.pres_pl_tokens = parser.pres_pl_tokens
        self.parts = parser.parts
        self.pasts = parser.pasts
        self.finite_tokens = parser.finite_tokens

    def density(self):
        """
        number of lexical tokens/number of tokens
        """
        return division(self.open_class_lemmas, self.lemmas)

    def LS(self):
        """
        number of sophisticated lexical tokens/number of lexical tokens
        """
        soph_lex_lemmas = [i for i in self.open_class_lemmas if i not in FIVE_T_FREQ_COCA]
        return division(soph_lex_lemmas, self.open_class_lemmas)

    def VS(self):
        """
        number of sophisticated verb lemmas/number of verb tokens
        """
        soph_verbs = set([i for i in verb_lemmas if i not in FREQ_VERBS_COCA_FROM_FIVE_T])
        VSI = division(soph_verbs, self.verb_lemmas)
        VSII = corrected_division(soph_verbs, self.verb_lemmas)
        VSIII = squared_division(soph_verbs, self.verb_lemmas)
        return VSI, VSII, VSIII

    def LFP(self):
        """
        Lexical Frequency Profile is the proportion of tokens:
        first - 1000 most frequent words
        second list - the second 1000
        third - University Word List (Xue & Nation 1989)
        none - list of those that are not in these lists
        """
        first = [i for i in lemmas if i in FIVE_T_FREQ_COCA[0:1000]]
        second = [i for i in lemmas if i in FIVE_T_FREQ_COCA[1000:2000]]
        third = [i for i in lemmas if i in UWL]
        first_procent = division(first, self.lemmas)
        second_procent = division(second, self.lemmas)
        third_procent = division(third, self.lemmas)
        none = 1 - (first_procent + second_procent + third_procent)
        return first_procent, second_procent, third_procent, none

    def NDW(self):
        """
        number of lemmas
        """
        return len(set(self.lemmas))

    def TTR(self):
        """
        number of lemmas/number of tokens
        """
        lemmas = set(self.lemmas)
        tokens = self.tokens
        TTR = division(lemmas, tokens)
        CTTR = corrected_division(lemmas, tokens)
        RTTR = root_division(lemmas, tokens)
        LogTTR = log_division(lemmas, tokens)
        Uber = uber(lemmas, tokens)
        return TTR, CTTR, RTTR, LogTTR, Uber

    def choose(self, n, k):
        """
        Calculates binomial coefficients
        """
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0

    def hyper(self, successes, sample_size, population_size, freq):
        """
        Calculates hypergeometric distribution
        """
        # probability a word will occur at least once in a sample of a particular size
        try:
            prob_1 = 1.0 - (float((self.choose(freq, successes) *
                                   self.choose((population_size - freq),
                                               (sample_size - successes)))) /
                            float(self.choose(population_size, sample_size)))
            prob_1 = prob_1 * (1 / sample_size)
        except ZeroDivisionError:
            prob_1 = 0
        return prob_1

    def D(self):
        prob_sum = 0.0
        tokens = self.tokens
        num_tokens = len(tokens)
        types_list = list(set(tokens))
        frequency_dict = collections.Counter(tokens)

        for items in types_list:
            # random sample is 42 items in length
            prob = self.hyper(0, 42, num_tokens, frequency_dict[items])
            prob_sum += prob

        return prob_sum

    def LV(self):
        """
        number of lexical lemmas/number of lexical tokens
        """
        lex_lemmas = set(self.lemmas)
        lex_tokens = self.tokens
        return len(lex_lemmas) / len(lex_tokens)

    def VV(self):
        """
        VVI: number of verb lemmas/number of verb tokens
        VVII: number of verb lemmas/number of lexical tokens
        """
        verb_lemmas = set(self.verb_lemmas)
        verb_tokens = self.verb_lemmas
        lex_tokens = self.open_class_lemmas
        VVI = division(verb_lemmas, verb_tokens)
        SVVI = squared_division(verb_lemmas, verb_tokens)
        CVVI = corrected_division(verb_lemmas, verb_tokens)
        VVII = division(verb_lemmas, lex_tokens)
        return VVI, SVVI, CVVI, VVII

    def NV(self):
        """
        number of noun lemmas/number of lexical tokens
        """
        noun_lemmas = set(self.noun_lemmas)
        lex_tokens = self.tokens
        return division(noun_lemmas, lex_tokens)

    def AdjV(self):
        """
        number of adjective lemmas/number of lexical tokens
        """
        adj_lemmas = set(self.get_adj_lemmas())
        lex_tokens = self.get_lex_lemmas()
        return self.division(adj_lemmas, lex_tokens)

    def AdvV(self):
        """
        number of adverb lemmas/number of lexical tokens
        """
        adv_lemmas = set(self.adv_lemmas)
        lex_tokens = self.open_class_lemmas
        return division(adv_lemmas, lex_tokens)

    def ModV(self):
        return self.AdjV() + self.AdvV()

    def one_random_list(self, l, length):
        result = []
        for i in range(length):
            random_element = random.choice(l)
            l.remove(random_element)
            result.append(random_element)
        return result, l

    def two_random_lists(self, l, length=10):
        list1, list2 = [], []
        if len(l) < length * 2:
            return self.two_random_lists(l, length=length - 1)
        else:
            list1, l = self.one_random_list(l, length)
            list2, l = self.one_random_list(l, length)
            return list1, list2

    def num_uniques(self, l):
        counter = collections.Counter(l)
        return list(counter.values()).count(1)

    def get_suffix(self, word):
        root = porter_stemmer.stem(word)
        suffix = word[len(root):]
        return suffix

    def get_suffixes(self):
        forms = self.tokens
        suffixes = [self.get_suffix(word) for word in forms]
        return list(filter(lambda s: s != '', suffixes))

    def derivational_suffixation(self):
        """
        number of suffixes on n's level/number of suffixes
        """
        suffixes = self.get_suffixes()
        level3_suffixes = [i for i in suffixes if i in suffixes_levels["level3"]]
        level4_suffixes = [i for i in suffixes if i in suffixes_levels["level4"]]
        level5_suffixes = [i for i in suffixes if i in suffixes_levels["level5"]]
        level6_suffixes = [i for i in suffixes if i in suffixes_levels["level6"]]
        der_suff3 = division(level3_suffixes, suffixes)
        der_suff4 = division(level4_suffixes, suffixes)
        der_suff5 = division(level5_suffixes, suffixes)
        der_suff6 = division(level6_suffixes, suffixes)
        return der_suff3, der_suff4, der_suff5, der_suff6

    def MCI(self):
        """
        MCI represents the average inflectional diversity for the parts of speech in the sample
        """
        verb_forms = self.verb_tokens
        suff_verb = [self.get_suffix(verb) for verb in verb_forms]
        list1, list2 = self.two_random_lists(suff_verb)
        diversity1 = len(set(list1))
        diversity2 = len(set(list2))
        mean_diversity = (diversity1 + diversity2) / 2
        num_uni = self.num_uniques(list1 + list2)
        IUV = num_uni / 2
        MCI = mean_diversity + IUV / 2 - 1
        return MCI

    def freq_finite_forms(self):
        """
        frequency of tensed(finite) forms
        """
        return division(self.finite_tokens, self.verb_tokens)

    def freq_aux(self):
        """
        frequency of modals(auxilaries)
        """
        return division(self.aux_forms, self.verb_tokens)

    def num_verb_forms(self):
        """
        number of different verb forms:
        infinitives, gerunds, present singular, present plural, past participle, past simple
        """
        return len(self.infinitive_tokens), len(self.gerund_tokens),\
               len(self.pres_sg_tokens), len(self.pres_pl_tokens), len(self.parts), len(self.pasts)

    def subfinder(self, mylist, pattern):
        matches = []
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                matches.append(pattern)
        return matches

    def num_dict_2_levels(self, d, prefix):
        num_all = 0
        result = {}
        for group in d:
            num_group = 0
            for subgroup in d[group]:
                num_subgroup = 0
                name_subgroup = list(subgroup.keys())[0]
                for word in list(subgroup.values())[0]:
                    num = len(re.findall(word.lower(), self.text.lower()))
                    num_all += num
                    num_subgroup += num
                    num_group += num
                    result[prefix + name_subgroup + "(" + word + ")"] = num
                result[prefix + name_subgroup] = num_subgroup
            result[prefix + group] = num_group
        result[prefix + 'all'] = num_all
        return result

    def num_linkings(self):
        """
        number of linking phrases (Swales & Feak 2009)
        """
        num_links_d = self.num_dict_2_levels(LINKINGS, 'link_')
        return num_links_d

    def num_4grams(self):
        """
        """
        num_all = 0
        for ngram in ngrams:
            num = len(self.subfinder([x.lower() if type(x) == str else x for x in self.get_forms()], ngram))
            num_all += num
        return num_all

    def num_func_ngrams(self):
        """
        number of linking phrases (Swales & Feak 2009)
        """
        num_links_d = self.num_dict_2_levels(FUNC_NGRAMS, '4grams_')
        return num_links_d
