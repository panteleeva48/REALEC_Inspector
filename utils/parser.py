from conllu import parse
from config import OPEN_CLASS


class ParserUDpipe:
    """Parses text using UDpipe."""

    def __init__(self):
        self.text = ''
        self.conllu = ''
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

    def text2conllu(self, text, model):
        self.text = text
        sentences = model.tokenize(self.text)
        for s in sentences:
            model.tag(s)
            model.parse(s)
        self.conllu = model.write(sentences, "conllu")

    def get_info(self):
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

        sentences = parse(self.conllu)
        for sentence in sentences:
            for token in sentence:
                lemma = token.get('lemma')
                form = token.get('form')
                self.lemmas.append(lemma)
                self.tokens.append(form)
                feats = token.get('feats')
                # todo: why method get does not work
                if not feats:
                    feats = {}
                if token.get('upostag') == 'VERB':
                    self.verb_lemmas.append(lemma)
                    self.verb_tokens.append(form)
                    if feats.get('VerbForm', '') == 'Fin':
                        self.finite_tokens.append(form)
                if token.get('upostag') == 'NOUN':
                    self.noun_lemmas.append(lemma)
                if token.get('upostag') == 'ADJ':
                    self.adj_lemmas.append(lemma)
                if token.get('upostag') == 'ADV':
                    self.adv_lemmas.append(lemma)
                if token.get('upostag') == 'AUX':
                    self.aux_forms.append(form)
                if token.get('upostag') in OPEN_CLASS:
                    self.open_class_lemmas.append(lemma)
                if feats.get('VerbForm', '') == 'Inf':
                    self.infinitive_tokens.append(form)
                if feats.get('VerbForm', '') == 'Ger':
                    self.gerund_tokens.append(form)
                # todo: check all forms here
                if feats.get('Person', '') == '3' and \
                        feats.get('Tense', '') == 'Pres' and\
                        feats.get('Mood', '') == 'Ind' and\
                        feats.get('VerbForm', '') == 'Fin':
                    if feats.get('Number', '') == 'Sing':
                        self.pres_sg_tokens.append(form)
                    if feats.get('Number', '') == 'Plur':
                        self.pres_pl_tokens.append(form)
                if feats.get('Tense', '') == 'Past' and feats.get('VerbForm', '') == 'Part':
                    self.parts.append(form)
                if feats.get('Mood', '') == 'Ind' and \
                        feats.get('Person', '') == '3' and\
                        feats.get('Tense', '') == 'Past' and\
                        feats.get('VerbForm', '') == 'Fin':
                    self.pasts.append(form)
