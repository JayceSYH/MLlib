import pandas as pd


class CartTree(object):
    def __init__(self, feature_name, split_value, df, branch_type="discrete"):
        self._type = branch_type
        self._feature_name = feature_name
        self._split_value = split_value
        self._nodes = {}

        if branch_type == "discrete":
            left_data = df[df[feature_name] == split_value]
            right_data = df[df[feature_name] != split_value]
            self._nodes['left'] = CartTreeNode(data=left_data, split_value=split_value)
            self._nodes['right'] = CartTreeNode(data=right_data, split_value=split_value)
        else:
            left_data = df[df[feature_name] <= split_value]
            right_data = df[df[feature_name] > split_value]
            self._nodes['left'] = CartTreeNode(data=left_data, split_value=split_value)
            self._nodes['right'] = CartTreeNode(data=right_data, split_value=split_value)

    def iternodes(self):
        return [self._nodes[k] for k in self._nodes]

    def clean_data(self):
        for node in self._nodes:
            node.clean_data()

    def predict(self, data_entry):
        if self._type == "discrete":
            if data_entry[self._feature_name] == self._split_value:
                sub_tree = self._nodes['left'].sub_tree
            else:
                sub_tree = self._nodes['right'].sub_tree
            if isinstance(sub_tree, CartTree):
                print("{0}({1})=>".format(self._feature_name, data_entry[self._feature_name]), end="")
                return sub_tree.predict(data_entry)
            else:
                print("{0}({1})=>{2}".format(self._feature_name, data_entry[self._feature_name], sub_tree))
                return sub_tree

        else:
            sub_tree = self._nodes["left" if data_entry[self._feature_name] <= self._split_value else "right"].sub_tree
            if isinstance(sub_tree, CartTree):
                return sub_tree.predict(data_entry)
            else:
                return sub_tree


class CartTreeNode(object):
    def __init__(self, data=None, split_value=None, sub_tree=None):
        self.sub_tree = sub_tree
        self.data = data
        self.split_value = split_value

    def set_sub_tree(self, tree):
        self.sub_tree = tree

    def clean_data(self):
        self.data = None