from parser import ParserUDpipe
from config import UDPIPE_MODEL

parser = ParserUDpipe()


def main(path, conllu_path, model):
    parser.parsing2conllu(path, conllu_path, model)


if __name__ == '__main__':
    path = '/Users/ira/Downloads/diplom/REALEC_Inspector/data/test.txt'
    conllu_path = '/Users/ira/Downloads/diplom/REALEC_Inspector/data/test.conllu'
    main(path, conllu_path, UDPIPE_MODEL)
