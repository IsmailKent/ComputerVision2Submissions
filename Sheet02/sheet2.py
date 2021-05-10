from RandomForest import Forest
from Sampler import PatchSampler
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
    cv2.imshow("train im", train_images[0])
    cv2.imshow("train seg", train_segmentations[0])
    cv2.imshow("test im", test_images[0])
    cv2.imshow("test seg", test_segmentations[0])
    
    sampler = PatchSampler(train_images, train_segmentations, range(4), 16)
    # list of (patch, class) pairs
    training_patches = sampler.extractpatches()
     
main()

# provide your implementation for the sheet 2 here


