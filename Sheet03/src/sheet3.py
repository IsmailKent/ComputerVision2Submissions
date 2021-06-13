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
path_train_2 = '../data/task_2_3_Data/train/'
path_test_2 = '../data/task_2_3_Data/test/'

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


# return 64x128 crop from center of image
def get_pos_crop(image):
    h , w , _ = image.shape
    
    start_w = w//2 - 64//2
    finish_w = w//2 + 64//2
    
    start_h = h//2 - 128//2
    finish_h = h//2 +  128//2
    
    return image[start_h:finish_h, start_w:finish_w]

# return 10 128x64 crops from random locations 
def get_neg_crops(image):
    crops = []
    h , w , _ = image.shape
    
    for _ in range(10):
    
        start1 = random.randint(0,h-128)
        start2 = random.randint(0,w-64)
        finish1= start1 + 128
        finish2 = start2 + 64
        
        crops.append(image[start1:finish1, start2:finish2])
    
    return crops

    
    # TODO: Create a HOG descriptor object to extract the features from the set of positive and negative samples 

    # positive samples: Get a crop of size 64*128 at the center of the image then extract its HOG features
    # negative samples: Sample 10 crops from each negative sample at random and then extract their HOG features
    # In total you should have  (x+10*y) training samples represented as HOG features(x=number of positive images, y=number of negative images),
    # save them and their labels in the path train_hog_path and train_labels in order to load them in section 3 


def task2():

    print('Task 2 - Extract HOG features')

    random.seed()
    np.random.seed()

    # Load image names
  
    filelist_train_pos = path_train_2 + 'filenamesTrainPos.txt'
    filelist_train_neg = path_train_2 + 'filenamesTrainNeg.txt'
    
    pos_file = open(filelist_train_pos,'r')
    neg_file = open(filelist_train_neg,'r')

    
    pos_HoGs = []
    neg_HoGs = []
    
    hog_dscr = cv.HOGDescriptor()
    hog_dscr2 = cv.HOGDescriptor()

    for i , line in enumerate(pos_file):
        image_name  = path_train_2 +'/pos/'+line.rstrip('\n') 
        img = cv.imread(image_name)
        crop = get_pos_crop(img)
        hog = hog_dscr.compute(crop)
        pos_HoGs.append(hog)
    pos_file.close()
    
    for i , line in enumerate(neg_file):
        # remove end of line char
        image_name  = path_train_2 +'/neg/'+line.rstrip('\n') 
        img = cv.imread(image_name)
        HoGs = []
        crops = get_neg_crops(img)
        for c in crops:


            hog = hog_dscr2.compute(c)
            HoGs.append(hog)

        neg_HoGs+=HoGs

        
    neg_file.close()
    
       
    print(len(pos_HoGs))
    print(len(neg_HoGs))
    
    hog_file = open(train_hog_path, "a")
    label_file = open(train_labels, "a")
    
    for hog in pos_HoGs:
        print(np.array2string(hog, separator=',',suppress_small=True), file=hog_file)
        print('1', file=label_file)

    
    for hog in neg_HoGs:
        print(np.array2string(hog, separator=',',suppress_small=True), file=hog_file)
        print('0', file=label_file)
    
    hog_file.close()
    label_file.close()
    
    







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
    #task1()

    # Task 2 - Extract HOG Features
    task2()

    # Task 3 - Train SVM
    #task3()

    # Task 5 - Multiple Detections
    #task5()

