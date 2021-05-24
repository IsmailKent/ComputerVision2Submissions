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
    
    segmentation_colors = np.array([[0,0,0],
                                    [255,0,0],
                                    [0,0,255],
                                    [0,255,0]])
    train_images , train_segmentations = read_images("images/train_images.txt")
    test_images , test_segmentations = read_images("images/test_images.txt")
    
    testing_image = cv2.imread('images/img_12.bmp')
    #testing_segmentation = cv2.imread('images/img_12_segMap.bmp')
    
    sampler = PatchSampler(train_images, train_segmentations, range(4), 16)
    # list of (patch, class) pairs
    training_patches , training_labels = sampler.extractpatches()
    
    tree_parameters = {'depth': 15 , 'minimum_patches_at_leaf':20, 'classes': range(4)}
    

    stand_alone_tree = DecisionTree(training_patches[:1000], training_labels[:1000], tree_parameters)
    stand_alone_tree.train()
    
    forest = Forest(training_patches[:5000], training_labels[:5000],tree_parameters,5)
    forest.create_forest()
    
    

    prediction_stand_alone_tree = stand_alone_tree.predict(testing_image)
    x ,y =   prediction_stand_alone_tree.shape
    prediction_stand_alone_tree_colored = np.zeros((x,y,3))
    for i in range (x):
        for j in range(y):
            prediction_stand_alone_tree_colored[i][j] = segmentation_colors[int(prediction_stand_alone_tree[i][j])]


    cv2.imshow("image",testing_image)
    
    cv2.imshow("tree_prediction.PNG", prediction_stand_alone_tree_colored)
    cv2.imwrite("tree_prediction.PNG", prediction_stand_alone_tree_colored)
    
    prediction_forest =forest.test(testing_image)
    
    x ,y = prediction_forest.shape
    prediction_forest_colored = np.zeros((x,y,3))
    
    for i in range (x):
        for j in range(y):
            prediction_forest_colored[i][j] = segmentation_colors[int(prediction_forest[i][j])]

    cv2.imshow("forest_prediction.PNG", prediction_forest_colored)
    cv2.imwrite("forest_prediction.PNG", prediction_forest_colored)
    

    cv2.waitKey(0)

    
    
    
    
     
    
main()

# provide your implementation for the sheet 2 here


