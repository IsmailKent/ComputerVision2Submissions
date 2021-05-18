from RandomForest import Forest
from Sampler import PatchSampler
from Tree import DecisionTree
import numpy as np
import cv2
import json


def read_images(filename):
     myfile = open(filename, "r")
     numbers_list = myfile.readline().split()
     numImages = int(numbers_list[0])
     numClasses = int(numbers_list[1])
     images = []
     segmentations = []
     for _ in range(numImages):
          line = myfile.readline().split()
          image_name = line[0]
          segmentation_name = line[1]
          image = cv2.imread("images/{}".format(image_name))
          segmentation = cv2.imread("images/{}".format(segmentation_name))
          images.append(image)
          segmentations.append(segmentation)
     myfile.close()   
     return images, segmentations


def main():
    train_images , train_segmentations = read_images("images/train_images.txt")
    test_images , test_segmentations = read_images("images/test_images.txt")

    
    sampler = PatchSampler(train_images, train_segmentations, range(4), 16)
    # list of (patch, class) pairs
    training_patches , training_labels = sampler.extractpatches()

    tree_parameters = {'depth': 20 , 'minimum_patches_at_leaf':20, 'classes': range(4)}
    
    stand_alone_tree = DecisionTree(training_patches[:1000], training_labels[:1000], tree_parameters)
    stand_alone_tree.train()
    
    prediction = stand_alone_tree.predict(train_images[3])
    cv2.imshow("image",train_images[3])
    cv2.imshow("prediction", prediction * 70)
    print(prediction)
    cv2.waitKey(0)

    
    
    
    
     
    
main()

# provide your implementation for the sheet 2 here


