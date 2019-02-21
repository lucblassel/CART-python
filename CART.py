import numpy as np
import pandas as pd



class Tree:

    def get_indent(level):
        return "    " * level

    def __init__(self, data, outcome, parent_node=None,
                 value="", min_samples_leaf=3):
        """
        Arguments:
            data {pd.DataFrame} -- data you want to train the tree on
            outcome {String} -- name of the column containing your target feature
            parent_node {Tree} -- parent node, if node is root then parent_node is None
            value {String} -- value of the of the parent split for this node
            min_samples_leaf {int} -- minimum number of cases in node
        """

        self.data = data
        self.outcome = outcome
        self.parent_node = parent_node
        self.value = value
        self.min_samples_leaf = min_samples_leaf
        self.child_nodes = []
        self.split_on = ""
        self.p_class = None

    def get_node_info(self):

        if self.parent_node is None:
            value = "root"
        else:
            value = "{} = {}".format(
                self.parent_node.split_on,
                self.value)

        if len(self.child_nodes) == 0:
            split = "{} = {}".format(
                self.outcome,
                self.p_class)
        else:
            split = "split on: {} into {} nodes".format(
                self.split_on,
                len(self.child_nodes))

        return "{}  -> {}".format(value, split)

    def print_tree(self, level=0):

        indent = Tree.get_indent(level)
        info = self.get_node_info()
        print(indent + info + '\n')

        if len(self.child_nodes) == 0:
            return

        for child in self.child_nodes:
            child.print_tree(level+1)

    def get_gini(self, subset):
        """takes the vector of targets as input

        Arguments:
            subset {pd.DataFrame} -- [description]

        Returns:
            [float] -- gini index for given data subset
        """

        proportions = subset[self.outcome].value_counts(normalize=True)
        return 1 - (proportions ** 2).sum()

    def get_feature_index(self, feature):
        """gets weighted gini index for a given feature

        Arguments:
            feature {String} -- feature for which to calculate the weighted index

        Returns:
            float -- weighted gini index
        """

        uniques = self.data[feature].value_counts(normalize=True).to_dict()
        ginis = {}
        for unique in uniques:
            gini = self.get_gini(self.data[self.data[feature] == unique])
            ginis[unique] = gini

        return np.sum([ginis[unique] * uniques[unique] for unique in uniques])

    def get_splits(self):
        features = self.data.columns.drop(self.outcome)

        for feature in features:
            if self.data[feature].

    def get_best_split(self):
        """return feature in the dataset for which the weighted gini index is the lowest

        Returns:
            String -- name of feature leading to best split
        """

        features = self.data.columns.drop(self.outcome)

        indexes = {feature: self.get_feature_index(feature)
                   for feature in features}

        best_feature = min(indexes, key=indexes.get)

        return best_feature

    def is_pure(self):
        """Checks if node is pure

        Returns:
            Bool -- purity of node
        """
        freqs = self.data[self.outcome].value_counts(normalize=True)

        return min(freqs) <= self.threshold

    def split(self):
        """Recursive function, that finds the best possible feature to split on in the dataset and creates a child node for each possible value of that feature.
        """

        if self.is_pure():  # stop condition
            print("node is pure")
            self.p_class = self.data[self.outcome].iloc[0]
            return

        best_feature = self.get_best_split()
        self.split_on = best_feature
        values = self.data[best_feature].unique()
        print("found best feature: {}".format(best_feature))
        print("split into: {}".format(values))

        for value in values:  # create splits
            child = Tree(
                self.data[self.data[best_feature] == value]
                    .drop(best_feature, axis=1),
                self.outcome,
                parent_node=self,
                value=value)
            self.child_nodes.append(child)
            child.split()

        return


if __name__ == "__main__":
    data = pd.read_csv('data.tab', sep='\t', header=0, index_col=0)
    tree = Tree(data, 'Decision')
    tree.split()
    tree.print_tree()
