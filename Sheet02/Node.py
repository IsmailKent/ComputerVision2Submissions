import numpy as np


class Node():
    def __init__(self):

        self.type = 'None'
        self.leftChild = -1
        self.rightChild = -1
        self.feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        self.probabilities = []

    # Function to create a new split node
    # provide your implementation
    def create_SplitNode(self, leftchild, rightchild, feature):
       node = Node()
       node.leftChild = leftchild
       node.rightChild = rightchild
       node.feature = feature
       return node

    # Function to create a new leaf node
    # provide your implementation
    def create_leafNode(self, labels, classes):
        node = Node()
        p = [None] * len(classes)
        for c in classes:
            p[c] = np.sum(labels==c)
        p /= np.sum(p)
        node.probabilities = p
        
        return node


    # feel free to add any helper functions