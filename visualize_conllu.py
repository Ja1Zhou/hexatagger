import sys
import spacy
# import spacy_conll
from conllu import parse
from spacy import displacy
from typing import TypedDict, List, Dict
class ConllUWord(TypedDict):
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str

class word_n_tag(TypedDict):
    text: str
    tag: str

class arc(TypedDict):
    start: int
    end: int
    label: str
    dir: str

class ConllUDoc(TypedDict):
    words: List[word_n_tag]
    arcs: List[arc]

# Load Spacy English Model
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("conll_formatter", last=True)

# Function to parse CoNLL-U file
def parse_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        return parse(data)

# Process parsed data
def get_word_n_tag(word: ConllUWord) -> word_n_tag:
    return word_n_tag({
        "text": word['form'],
        "tag": word['upos']
    })

def get_arc(word: ConllUWord) -> arc:
    head = word['head'] - 1
    tail = word['id'] - 1
    start, end = (head, tail) if head < tail else (tail, head)
    dir = "left" if head > tail else "right"
    return arc({
        "start": start,
        "end": end,
        "label": word['deprel'],
        "dir": dir
    })

def process_data(parsed_data: List[ConllUWord]):
    '''
    {
        "words": [
            {"text": "This", "tag": "DT"},
            {"text": "is", "tag": "VBZ"},
            {"text": "a", "tag": "DT"},
            {"text": "sentence", "tag": "NN"}
        ],
        "arcs": [
            {"start": 0, "end": 1, "label": "nsubj", "dir": "left"},
            {"start": 2, "end": 3, "label": "det", "dir": "left"},
            {"start": 1, "end": 3, "label": "attr", "dir": "right"}
        ]
    }
    '''
    words = list(map(get_word_n_tag, parsed_data))
    not_head = [word for word in parsed_data if word['head'] != 0]
    arcs = list(map(get_arc, not_head))
    return ConllUDoc({
        "words": words,
        "arcs": arcs
    })
    
# File path to your CoNLL-U file
# file_path = 'data/garden_path_more.conll'
file_path = 'data/eval_pos.conll'
if len(sys.argv) > 1:
    file_path = sys.argv[1]
# Parse and process the file
parsed_data = parse_conllu(file_path)
docs = list(map(process_data, parsed_data))

# Display dependency tree
displacy.serve(docs, style="dep", manual=True, auto_select_port=True)
# for doc in docs:
#     try:
#         displacy.serve(doc, style="dep", manual=True, auto_select_port=True)
#     except:
#         print(doc)
        