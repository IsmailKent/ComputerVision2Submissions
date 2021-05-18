import numpy as np
from Node import Node
from Sampler import PatchSampler


class DecisionTree():
    def __init__(self, patches, labels, tree_param):

        self.patches, self.labels = patches, labels
        self.depth = tree_param['depth']
        #self.pixel_locations = tree_param['pixel_locations']
        #self.random_color_values = tree_param['random_color_values']
        #self.no_of_thresholds = tree_param['no_of_thresholds']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.nodes = []
        
        self.generate_random_pixel_location()
        self.generate_random_threshold_values()
        self.generate_random_color_values()
        
        # since for some reason node creation is instance functions
        self.creator = Node()
        self.root = None
        
    # Function to train the tree
    # provide your implementation
    # should return a trained tree with provided tree param
    def train(self):
        # to index with lists of indices
        patches = np.array(self.patches)
        labels = np.array(self.labels)
        self.root = self.train_helper(patches,labels, self.depth)
        return self.root
    
    def train_helper(self, patches, labels, depth):
        if (depth==0):
            return self.creator.create_leafNode(labels,self.classes)
        
        best_left,best_right, color, pixel_location, th = self.best_split(patches, labels)

        if (len(best_left) < self.minimum_patches_at_leaf or len(best_right) < self.minimum_patches_at_leaf):
             return self.creator.create_leafNode(labels,self.classes)
         
        left_node = self.train_helper(patches[best_left], labels[best_left], depth-1)
        right_node =  self.train_helper(patches[best_right], labels[best_right], depth-1)
        feature = {'color': color, 'pixel_location': pixel_location , 'th':th}
        node = self.creator.create_SplitNode(left_node , right_node, feature)
        self.nodes.append(node)
        print("Created Node with feature: {}".format(node.feature))
        return node
        
        

    # Function to predict probabilities for single image
    # provide your implementation
    # should return predicted class for every pixel in the test image
    def predict(self, I):
        patch_size = self.patches[0].shape[0]
        predictions = np.zeros((I.shape[0] - patch_size, I.shape[1] - patch_size))

        for i in range(I.shape[0] - patch_size):
            for j in range(I.shape[1] - patch_size):
                patch = I[i:i+patch_size,j:j+patch_size]
                prediction = self.traverse_tree(self.root, patch)
                predictions[i][j]=prediction
                     
        return predictions
    
    def traverse_tree(self,node, patch):
        # if node is leaf node return value
        
        if (node.leftChild == -1):
            print("predicted ",np.argmax(node.probabilities))
            return  np.argmax(node.probabilities)
        
        response = self.getFeatureResponse([patch], node.feature)[0]
        if (response):
            return self.traverse_tree(node.rightChild, patch)
        return self.traverse_tree(node.leftChild, patch)
        
        

    # Function to get feature response for a random color and pixel location
    # provide your implementation
    # should return feature response for all input patches
    def getFeatureResponse(self, patches, feature):
        
        #from node class
        # feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}

        x , y = feature["pixel_location"]
        color = feature["color"]
        threshold = feature["th"]
        responses = []
        for patch in patches:
            responses.append(patch[x][y][color] > threshold)
            
        return responses

    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        pass

    # Function to get a random pixel location
    # provide your implementation
    # should return a random location inside the patch
    def generate_random_pixel_location(self):
        patch_size = self.patches[0].shape[0]
        random_locations = []
        # 100
        while(len(random_locations) < 100):
            random_locations.append( ( np.random.randint(0,patch_size) , np.random.randint(0,patch_size) ))
        self.pixel_locations = random_locations
        
    def generate_random_threshold_values(self):
        threshold_values = []
        # 40
        while (len(threshold_values)<50):
            threshold_values.append( np.random.randint(0,256))
        self.no_of_thresholds = threshold_values
        
    def generate_random_color_values(self):
        color_values = []
        #10
        while (len(color_values)<10):
            color_values.append(np.random.randint(0,3))
        self.random_color_values = color_values
         

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, labels):
        l = np.array(labels)
        n = l.size
        if (n==0):
            return 0
        classes = self.classes
        Sum =0
        for c in classes:
            p = np.sum(l == c) / n
            if (p!=0):
                Sum-= p * np.log2(p)
                
        return Sum

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        return EntropyAll - Entropyleft * Nleft / Nall - Entropyright * Nright / Nall

    # Function to get the best split for given patches with labels
    # provide your implementation
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        print("splitting")
        best_gain = -1
        best_split_feature =  {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        entropyAll = self.compute_entropy(labels)
        best_left = None
        best_right = None
        for c in self.random_color_values:
            for x,y in self.pixel_locations:
                for th in self.no_of_thresholds:
                    #from node class
                    # feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
                    feature = {'color': c, 'pixel_location': [x,y] , 'th':th}
                    responses = self.getFeatureResponse(patches,feature)
                    indices_left = [idx for idx, response in enumerate(responses) if not response]
                    indices_right = [idx for idx, response in enumerate(responses) if response]
                    labels_left = np.array(labels)[indices_left]
                    labels_right = np.array(labels)[indices_right]
                    entropyLeft = self.compute_entropy(labels_left)
                    entropyRight = self.compute_entropy(labels_right)
                    gain = self.get_information_gain(entropyLeft,entropyRight,entropyAll, len(labels), len(labels_left), len(labels_right))
                    if (gain>best_gain):
                        best_gain = gain
                        best_split_feature = feature
                        best_left = indices_left
                        best_right = indices_right
                        
                            
        return best_left,best_right, best_split_feature['color'], best_split_feature['pixel_location'], best_split_feature['th']
    # feel free to add any helper functions

