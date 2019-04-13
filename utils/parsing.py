import re
import pandas as pd
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

from model import Model
model = Model('english-partut-ud-2.3-181115.udpipe')

class ParserUDpipe:
    """Parses text using udpipe."""
 
    def __init__(self, text):
        self.text = text
        
    def parsing(self):
        sentences = model.tokenize(self.text)
        for s in sentences:
            model.tag(s)
            model.parse(s)
        conllu = model.write(sentences, "conllu")
        return conllu
    
    def conllu2df(self, sentences=False):
        conllu = self.parsing()
        sents = re.findall('# text = (.+)\n', conllu)
        conllu = re.sub('# .+?\n', '', conllu)
        sent_lst = re.findall('(1\t.+?)\n\n', conllu, re.DOTALL)
        dfs = []
        for each in sent_lst:
            rows = each.split('\n')
            string = 'Id\tForm\tLemma\tUPosTag\tXPosTag\tFeats\tHead\tDepRel\tDeps\tMisc\n'
            string += "\n".join(rows)
            string = StringIO(string)
            df = pd.read_csv(string, sep="\t")
            try:
                df = df[~df.Id.str.contains("-")]
            except:
                pass
            dfs.append(df)
        if sentences:
            return sents, dfs
        else:
            df_all = pd.concat(dfs, ignore_index=True)
            return df_all
    
    def get_sentances(self):
        sentances = self.conllu2df(sentences=True)
        return sentances