import cv2 as cv
import numpy as np
import random

from custom_hog_detector import Custom_Hog_Detector
# Global constants

# crop/patch dimensions for the training samples
width = 64
height = 128

num_negative_samples = 10 # number of negative samples per image
train_hog_path = '../train_hog_descs.dat' # the file to which you save the HOG descriptors of every patch
train_labels = '../labels_train.dat' # the file to which you save the labels of the training data
my_svm_filename = '../my_pretrained_svm.dat' # the file to which you save the trained svm 

#data paths
test_images_1 = '../data/task_1_testImages/'
path_train_2 = '../data/task_2_3_Data/01Train/'
path_test_2 = '../data/task_2_3_Data/02Test/'

#***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding box of the detections (people)
# returns None

def drawBoundingBox(im, detections):
    draw_detections(im,detections)

# helper functions from peopledetect.py
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
        
        
def task1():
    print('Task 1 - OpenCV HOG')
    #source used peopledetect.py
    
    
    # Load images

    filelist = test_images_1 + 'filenames.txt'
    

    # TODO: Create a HOG descriptor object to extract the features and detect people. Do this for every 
    #       image, then draw a bounding box and display the image
    
    #get the already trained people detector:
    hog = cv.HOGDescriptor()
    hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )
    
    # load images from file
    
    
    images_file = open(filelist,'r')
    images_list = []
    for line in images_file:
            # remove end of line char
            line = line.rstrip('\n') 
            image_name = line.split('/')[1]
            image_path = test_images_1+'/'+image_name
            image = cv.imread(image_path)
            images_list.append(image)
            
    images_file.close()

    for img in images_list:
        found, _w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        print('%d (%d) found' % (len(found_filtered), len(found)))
        cv.imshow('img', img)
        ch = cv.waitKey()
        if ch == 27:
            break

    print('Done')


    
    cv.destroyAllWindows()
    cv.waitKey(0)


def task2():

    print('Task 2 - Extract HOG features')

    random.seed()
    np.random.seed()

    # Load image names
  
    filelist_train_pos = path_train_2 + 'filenamesTrainPos.txt'
    filelist_train_neg = path_train_2 + 'filenamesTrainNeg.txt'
    # TODO: Create a HOG descriptor object to extract the features from the set of positive and negative samples 

    # positive samples: Get a crop of size 64*128 at the center of the image then extract its HOG features
    # negative samples: Sample 10 crops from each negative sample at random and then extract their HOG features
    # In total you should have  (x+10*y) training samples represented as HOG features(x=number of positive images, y=number of negative images),
    # save them and their labels in the path train_hog_path and train_labels in order to load them in section 3 







def task3(): 
    print('Task 3 - Train SVM and predict confidence values')
      #TODO Create 3 SVMs with different C values, train them with the training data and save them
      # then use them to classify the test images and save the results
    

    filelist_testPos = path_test_2 + 'filenamesTestPos.txt'
    filelist_testNeg = path_test_2 + 'filenamesTestNeg.txt'
    




def task5():

    print ('Task 5 - Eliminating redundant Detections')
    

    # TODO: Write your own custom class myHogDetector 
      
    my_detector = Custom_Hog_Detector(my_svm_filename)
   
    # TODO Apply your HOG detector on the same test images as used in task 1 and display the results

    print('Done!')
    cv.waitKey()
    cv.destroyAllWindows()






if __name__ == "__main__":

    # Task 1 - OpenCV HOG
    task1()

    # Task 2 - Extract HOG Features
    #task2()

    # Task 3 - Train SVM
    #task3()

    # Task 5 - Multiple Detections
    #task5()

