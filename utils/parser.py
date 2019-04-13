class ParserUDpipe:
    """Parses text using UDpipe."""

    def __init__(self):
        self.text = ''

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def write_file(self, conllu_path, conllu):
        with open(conllu_path, 'w') as file:
            file.write(conllu)

    def parsing2conllu(self, path, conllu_path, model):
        print('Parsing', path)
        self.text = self.read_file(path)
        sentences = model.tokenize(self.text)
        for s in sentences:
            model.tag(s)
            model.parse(s)
        conllu = model.write(sentences, "conllu")
        print('Write into', conllu_path)
        self.write_file(conllu_path, conllu)
