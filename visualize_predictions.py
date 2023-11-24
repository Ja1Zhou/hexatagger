from spacy import displacy
from typing import TypedDict, List, Dict, Tuple
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

class TokenWithId(TypedDict):
    id: int
    token: str

class HexaTag(TypedDict):
    tag: str
    hexa: str

class Node(TypedDict):
    is_leaf: bool
    head_id: int
class BinaryHeadedTreeInnerNode(Node):
    parent: Node
    hexa_tag: HexaTag
    left_child: Node
    right_child: Node
class BinaryHeadedTreeLeafNode(Node):
    token_with_id: TokenWithId
    tag: str

def get_hexa_n_tag(one_tag:str) -> HexaTag:
    hexa, tag = one_tag.split('/')
    if hexa.isupper():
        tag = tag[-1]
    return HexaTag(tag=tag, hexa=hexa)

def parse_hexa_outputs(paired_output: Tuple[str, List[str]]) -> ConllUDoc:
    '''
    input: ('The old man the boat .', ['l/det', 'L/X^^^1', 'r/nsubj', 'L/X^^^0', 'r/root', 'L/X^^^0', 'l/det', 'R/X^^^1', 'r/dep', 'L/X^^^0', 'r/punct'])
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
    tokens = paired_output[0].split()
    # token with id
    words = [word_n_tag(text=token, tag=tag.split('/')[1]) for token, tag in zip(tokens, paired_output[1][::2])]
    idx_to_words = {i: word for i, word in enumerate(words)}
    tokens = [TokenWithId(id=i, token=token) for i, token in enumerate(tokens)]
    hexa_tags = list(map(get_hexa_n_tag, paired_output[1]))
    stack:List[Node] = []
    for hexa_tag in hexa_tags:
        if hexa_tag['hexa'] == 'l':
            # shift onto stack
            token = tokens.pop(0)
            stack.append(BinaryHeadedTreeLeafNode(is_leaf=True, head_id=token['id'], token_with_id=token, tag=hexa_tag['tag']))
        elif hexa_tag['hexa'] == 'r':
            # right child
            token = tokens.pop(0)
            right_child = BinaryHeadedTreeLeafNode(is_leaf=True, head_id=token['id'], token_with_id=token, tag=hexa_tag['tag'])
            parent = stack.pop()
            rightmost_child = parent
            while rightmost_child['right_child'] is not None:
                rightmost_child = rightmost_child['right_child']
            rightmost_child['right_child'] = right_child
            right_child['parent'] = rightmost_child
            stack.append(parent)
        elif hexa_tag['hexa'] == 'L':
            # create inner node
            left_child = stack.pop()
            parent = BinaryHeadedTreeInnerNode(is_leaf=False, head_id=left_child['head_id'], hexa_tag=hexa_tag, left_child=left_child, right_child=None)
            left_child['parent'] = parent
            stack.append(parent)
        elif hexa_tag['hexa'] == 'R':
            left_child = stack.pop()
            parent = BinaryHeadedTreeInnerNode(is_leaf=False, head_id=left_child['head_id'], hexa_tag=hexa_tag, left_child=left_child, right_child=None)
            left_child['parent'] = parent
            grand_parent = stack.pop()
            rightmost_child = grand_parent
            while rightmost_child['right_child'] is not None:
                rightmost_child = rightmost_child['right_child']

            rightmost_child['right_child'] = parent
            parent['parent'] = rightmost_child
            stack.append(grand_parent)
    arcs = []
    dfs_traversal(stack[0], arcs, idx_to_words)
    # print(arcs)
    return ConllUDoc(words=words, arcs=arcs)

def dfs_traversal(node: Node, arcs:List[arc], idx_to_words: Dict[int, word_n_tag]):
    if node['is_leaf']:
        return
    # must has left and right child
    new_node: BinaryHeadedTreeInnerNode = node
    dfs_traversal(new_node['left_child'], arcs, idx_to_words)
    dfs_traversal(new_node['right_child'], arcs, idx_to_words)
    new_node['head_id'] = new_node['left_child']['head_id'] if new_node['hexa_tag']['tag'] == '0' else new_node['right_child']['head_id']
    start = new_node['left_child']['head_id']
    end = new_node['right_child']['head_id']
    dir = 'left' if new_node['hexa_tag']['tag'] == '1' else 'right'
    label_idx = new_node['left_child']['head_id'] if new_node['hexa_tag']['tag'] == '1' else new_node['right_child']['head_id']
    label = idx_to_words[label_idx]['tag']
    arcs.append(arc(start=start, end=end, label=label, dir=dir))

# output_file = 'outputs/garden_path_output.txt'
# output_file = 'outputs/garden_path_final_output.txt'
output_file = 'outputs/garden_path_final_with_pos_output.txt'
# output_file = 'outputs/infer_output.txt'
with open(output_file, 'r') as f:
    whole_output = f.read().splitlines()

# identify the first line with '-' * 10
for i, line in enumerate(whole_output):
    if line == '-' * 10:
        break
whole_output = whole_output[i+1:-1]
# filter out lines with '-' * 10
whole_output = [line for line in whole_output if line != '-' * 10]
# remove lines with index % 3 == 2
whole_output = [line for i, line in enumerate(whole_output) if i % 3 != 2]
# eval lines with index % 2 == 1
whole_output = [eval(line) if i % 2 == 1 else line for i, line in enumerate(whole_output) ]
# create pair for every two lines
# eg: ('The old man the boat .', ['l/det', 'L/X^^^1', 'r/nsubj', 'L/X^^^0', 'r/root', 'L/X^^^0', 'l/det', 'R/X^^^1', 'r/dep', 'L/X^^^0', 'r/punct'])
paired_output = [(whole_output[i], whole_output[i+1]) for i in range(0, len(whole_output), 2)]
docs = list(map(parse_hexa_outputs, paired_output))
displacy.serve(docs, style="dep", manual=True, auto_select_port=True)
# doc = parse_hexa_outputs(paired_output[0])
# displacy.serve(doc, style="dep", manual=True, auto_select_port=True)