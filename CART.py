import numpy as np

class Tree:

    def __init__(self, parent_node, data):
        self.parent = parent_node
        self.data = data

    def calculate_gini(outcomes):
        """takes the vector of targets as input
        
        Arguments:
            outcomes {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        proportions = outcomes.value_counts(normalize=True)
        return 1 - (proportions ** 2).sum()

    def get_best_split()