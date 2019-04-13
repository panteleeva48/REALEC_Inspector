from conllu import parse
from config import OPEN_CLASS


class ParserUDpipe:
    """Parses text using UDpipe."""

    def __init__(self):
        pass
        self.text = ''
        self.conllu = ''
        self.lemmas = []
        self.tokens = []
        self.verb_lemmas = []
        self.noun_lemmas = []
        self.adj_lemmas = []
        self.adv_lemmas = []
        self.open_class_lemmas = []

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
        sentences = parse(self.conllu)
        for sentence in sentences:
            for token in sentence:
                self.lemmas.append(token['lemma'])
                self.tokens.append(token['form'])
                if token['upostag'] == 'VERB':
                    self.verb_lemmas.append(token['lemma'])
                if token['upostag'] == 'NOUN':
                    self.noun_lemmas.append(token['lemma'])
                if token['upostag'] == 'ADJ':
                    self.adj_lemmas.append(token['lemma'])
                if token['upostag'] == 'ADV':
                    self.adv_lemmas.append(token['lemma'])
                if token['upostag'] in OPEN_CLASS:
                    self.open_class_lemmas.append(token['lemma'])
