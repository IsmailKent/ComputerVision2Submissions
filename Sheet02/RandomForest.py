from Tree import DecisionTree
import numpy as np
import json


class Forest():
    def __init__(self, patches=[], labels=[], tree_param=[], n_trees=1):

        self.patches, self.labels = patches, labels
        self.tree_param = tree_param
        self.ntrees = n_trees
        self.trees = []
        # This part is useless here, because we don't want to train them all with same patches
        # and get practically the same tree n times
        #for i in range(n_trees):
        #   self.trees.append(DecisionTree(self.patches, self.labels, self.tree_param))

    # Function to create ensemble of trees
    # provide your implementation
    # Should return a trained forest with n_trees
    def create_forest(self):
        
        # since the sampler already shuffles the data, there is no need for random selection   
        interval_length = len(self.patches) // self.ntrees
        i=0
        for _ in range(self.ntrees):
            end = min(i+interval_length , len(self.patches))
            interval = self.patches[i:end]
            labels = self.labels[i:end]
            tree = DecisionTree(interval, labels, self.tree_param)
            tree.train()
            self.trees.append(tree)
            i+=interval_length
            
            
        
        return self

    # Function to apply the trained Random Forest on a test image
    # provide your implementation
    # should return class for every pixel in the test image
    def test(self, I):
        
        # loop over patches in I
        
        patch_size = self.patches[0].shape[0]
        prediction =  np.zeros((I.shape[0] - patch_size, I.shape[1] - patch_size))
        
        predictions = np.zeros(( self.ntrees, I.shape[0] - patch_size, I.shape[1] - patch_size  ))

        for i in range (self.ntrees):
            p = self.trees[i].predict(I)
            predictions[i] = p
        
        
        
        # take majority vote
        for x in range(prediction.shape[0]):
            for y in range(prediction.shape[1]):
                nclasses = len(self.trees[0].classes)
                vote = [0] * nclasses
                for i in range(self.ntrees):
                    vote[int(predictions[i][x][y])]+=1
                prediction[x][y] = np.argmax(vote)
        return prediction

    # feel free to add any helper functions
