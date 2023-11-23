from abc import ABC

from nltk import ParentedTree as PTree
from nltk import Tree

from const import DUMMY_LABEL
from tagging.tetratagger import BottomUpTetratagger
from tagging.transform import RightCornerTransformer
from tagging.tree_tools import debinarize_lex_tree, expand_unary


class HexaTagger(BottomUpTetratagger, ABC):
    def preprocess(self, original_tree: Tree) -> PTree:
        tree = original_tree.copy(deep=True)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)

        ptree = PTree.convert(tree)
        root_label = ptree.label()
        tree_lc = PTree(root_label, [])
        RightCornerTransformer.transform(tree_lc, ptree, ptree)
        return tree_lc

    @staticmethod
    def create_shift_tag(label: str, left_or_right: str) -> str:
        arc_label = label.split("^^^")[-1]
        arc_label = arc_label.split("+")[0]
        return left_or_right + "/" + arc_label

    @staticmethod
    def _create_bi_reduce_tag(label: str, left_or_right: str) -> str:
        label = label.split("\\")[1]
        head_idx = label.split("^^^")[-1]
        # label = label.split("^^^")[0]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return f'{left_or_right}' + "/" + f"{DUMMY_LABEL}^^^{head_idx}"
        else:
            return f'{left_or_right}' + "/" + label.replace("+", "/")

    @staticmethod
    def _create_unary_reduce_tag(label: str, left_or_right: str) -> str:
        label = label.split("\\")[0]
        head_idx = label.split("^^^")[-1]
        # label = label.split("^^^")[0]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return f'{left_or_right}' + f"/{DUMMY_LABEL}^^^{head_idx}"
        else:
            return f'{left_or_right}' + "/" + label

    def tags_to_tree_pipeline(self, tags: [str], input_seq: []) -> Tree:
        ptree = self.tags_to_tree(tags, input_seq)
        return self.postprocess(ptree)

    @staticmethod
    def _create_pre_terminal_label(tag: str, default="X") -> str:
        arc_label = tag.split("/")[1]
        return f"X^^^{arc_label}+"

    @staticmethod
    def _create_unary_reduce_label(tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            return DUMMY_LABEL
        return tag[idx + 1:].replace("/", "+")

    @staticmethod
    def _create_reduce_label(tag: str) -> str:
        '''
        Will be called dynamically.
        '''
        idx = tag.find("/")
        if idx == -1:
            label = "X\\|"  # to mark the second part as an extra node created via binarizaiton
        else:
            label = "X\\" + tag[idx + 1:].replace("/", "+")
        return label

    def postprocess(self, transformed_tree: PTree) -> Tree:
        tree = PTree("X", ["", ""])
        tree = RightCornerTransformer.rev_transform(tree, transformed_tree)
        tree = Tree.convert(tree)
        if len(tree.leaves()) == 1:
            expand_unary(tree)
            # edge case with one node
            return tree

        debinarized_tree = Tree(tree.label(), [])
        debinarize_lex_tree(tree, debinarized_tree)
        expand_unary(debinarized_tree)
        # debinarized_tree.pretty_print()
        return debinarized_tree

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        created_node_stack = []
        node = None
        # expands tags as r leaf as a right inner node and a left leaf
        expanded_tags = self.expand_tags(tags)
        # print sentence
        print('-' * 10)
        print(' '.join([x[0] for x in input_seq]))
        print('-' * 10)
        print(tags)
        print('-' * 10)
        if len(expanded_tags) == 1:  # base case
            assert expanded_tags[0].startswith('l')
            prefix = self._create_pre_terminal_label(expanded_tags[0], "")
            return PTree(prefix + input_seq[0][1], [input_seq[0][0]])
        for tag in expanded_tags:
            if tag.startswith('l'):  # shift
                # something like X^^^nn+
                prefix = self._create_pre_terminal_label(tag, "")
                # input_seq[0][1] will be X^^^nn+NNP, placeholder
                # input_seq[0][0] will be the word
                created_node_stack.append(PTree(prefix + input_seq[0][1], [input_seq[0][0]]))
                input_seq.pop(0)
            else:
                node = PTree("X", [])
                if tag.startswith('R'):  # normal reduce
                    # left child
                    last_node = created_node_stack.pop()
                    # parent
                    last_2_node = created_node_stack.pop()
                    # should create a node with tag
                    # left child should be the first one poped
                    # parent should be the second one poped
                    # should replace dummy node in parent with current node
                    created_node_stack.append(self._reduce(node, last_node, last_2_node, tag))
                elif tag.startswith('L'):  # unary reduce
                    # should create a node with tag
                    # left child should be the one poped
                    # right child should be set to dummy
                    created_node_stack.append(
                        self._unary_reduce(node, created_node_stack.pop(), tag))
        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node

    def _reduce(self, node, last_node, last_2_node, tag):
        # parent node is set with label??
        # parent as left child??
        # right child is supposed left child??
        label = self._create_reduce_label(tag)
        last_2_node.set_label(label)
        node.insert(0, last_2_node)
        node.insert(1, last_node)
        return node
    
    def _unary_reduce(self, node, last_node, tag):
        # left child is with label + dummy??
        # right child is supposed left child??
        label = self._create_unary_reduce_label(tag)
        node.insert(0, PTree(label + "\\" + label, ["EPS"]))
        node.insert(1, last_node)
        return node