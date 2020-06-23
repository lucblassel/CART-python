#! /usr/bin/env python3

import pandas as pd
from pandas.api.types import is_categorical, is_string_dtype, is_bool
from sklearn.datasets import load_iris


def check_if_categorical(feature):
    return (
        is_categorical(feature) or
        is_string_dtype(feature) or
        is_bool(feature)
        )


class Node:

    def get_indent(level):
        return "    " * level

    def __init__(self, data, target,
                 value="", min_samples_leaf=3, level=1, max_depth=3):
        """initializes a Node object

        Arguments:
            data {pd.DataFrame} -- training data for the tree
            target {str} -- name of the target feature

        Keyword Arguments:
            value {str} -- value to display for node (default: {""})
            min_samples_leaf {int} -- minimum number of examples at a leaf node (default: {3})
            level {int} -- depth level of node in the tree (default: {1})
            max_depth {int} -- maximum depth of the tree (default: {3})
        """

        self.data = data
        self.target = target
        self.value = value
        self.min_samples_leaf = min_samples_leaf
        self.left = None
        self.right = None
        self.prediction = None
        self.level = level
        self.max_depth = max_depth
        self.leaves = []

    def display(self):
        lines, _, _, _ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self): # stolen from https://stackoverflow.com/a/54074933/8650928
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.value
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.value
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def get_gini(self, subset):
        """takes the vector of targets as input

        Arguments:
            subset {pd.DataFrame} -- the left side of the split data

        Returns:
            [float] -- gini index for given data subset
        """

        proportions = subset[self.target].value_counts(normalize=True)
        return 1 - (proportions ** 2).sum()

    def get_delta_i(self, subset):
        """gets the delta i for a given split

        Arguments:
            subset {pd.DataFrame} -- the left side of the split data

        Returns:
            [float] -- delta i for this split
        """

        gini = self.get_gini(self.data)

        left = subset
        right = self.data.drop(subset.index, axis=0)

        p_left = len(left) / len(self.data)
        p_right = 1 - p_left

        sub_left = p_left * self.get_gini(left)
        sub_right = p_right * self.get_gini(right)

        return gini - sub_left - sub_right

    def get_categorical_splits(self, feature):
        splits = {}
        for unique in self.data[feature].unique():
            splits[(feature, unique, 'categorical')] = self.data[
                self.data[feature] == unique]
        return splits

    def get_numerical_splits(self, feature):
        splits = {}
        uniques = self.data[feature].unique()
        for value in uniques:
            if value != max(uniques):
                splits[(feature, value, 'numerical')] = self.data[
                    self.data[feature] <= value]
        return splits

    def get_splits(self):
        features = self.data.columns.drop(self.target)
        all_splits = {}

        for feature in features:

            if check_if_categorical(self.data[feature]):
                all_splits.update(self.get_categorical_splits(feature))
            else:
                all_splits.update(self.get_numerical_splits(feature))

        return all_splits

    def get_best_split(self):
        all_splits = self.get_splits()
        delta_is = {}

        for key, split in all_splits.items():
            delta_is[key] = self.get_delta_i(split)

        return max(delta_is, key=delta_is.get)

    def is_pure(self):
        return len(self.data[self.target].unique()) == 1

    def too_small(self):
        return len(self.data) <= self.min_samples_leaf

    def too_deep(self):
        return self.level >= self.max_depth

    def no_splits(self):
        return self.get_splits() == {}

    def split(self):
        """Recursive function, that finds the best possible feature to split on in the dataset and creates a child node for each possible value of that feature.
        """

        if (self.is_pure() or self.too_deep() or
            self.no_splits() or self.too_small()):  # stop condition

            self.prediction = self.data[self.target].value_counts().idxmax()
            self.value = ' ({})'.format(
                self.prediction)
            return

        best_split = self.get_best_split()

        self.split_feature = best_split[0]
        self.split_value = best_split[1]
        self.split_type = best_split[2]

        if self.split_type == 'categorical':
            left_data = self.data[
                self.data[self.split_feature] == self.split_value]
            right_data = self.data[
                self.data[self.split_feature] != self.split_value]
            self.value = "{} = {}".format(
                self.split_feature, self.split_value
            )

        elif self.split_type == 'numerical':
            left_data = self.data[
                self.data[self.split_feature] <= self.split_value
            ]
            right_data = self.data[
                self.data[self.split_feature] > self.split_value
            ]
            self.value = "{} <= {}".format(
                self.split_feature, self.split_value
            )
        else:
            raise ValueError(
                'splits can be either numerical or categorical'
                )

        child_params = {
            'target': self.target,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'level': self.level + 1
        }

        self.left = Node(left_data, **child_params)
        self.right = Node(right_data, **child_params)

        self.left.split()
        self.right.split()

        return


    def get_leaves(self):

        if self.left is None and self.right is None:
            return [self]
        if self.leaves != []:
            return self.leaves

        if self.left is not None:
            self.leaves.extend(self.left.get_leaves())
        if self.right is not None:
            self.leaves.extend(self.right.get_leaves())

        return self.leaves

    def count_leaves(self):
        self.get_leaves()
        return len(self.leaves)



if __name__ == "__main__":
    data_mixed = pd.read_csv('data_mixed.csv', header=0, index_col=0)

    tree2 = Node(data_mixed, 'play', max_depth=4)
    tree2.split()
    tree2.display()

    iris = load_iris()
    iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    iris_df['species'] = iris['target']
    iris_df['species'] = iris_df['species'].apply(lambda i: iris['target_names'][i])

    tree_iris = Node(iris_df, 'species', max_depth=4)
    tree_iris.split()
    tree_iris.display()

