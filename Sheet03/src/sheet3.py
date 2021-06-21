import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

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
    
    
    return(pos_HoGs,neg_HoGs )



def plot_results( res_pos_svm ,res_neg_svm, title):
   labels_pos = np.ones(len(res_pos_svm))
   #dif_pos = abs(labels_pos -  res_pos_svm)
   labels_neg = -1* np.ones(len(res_neg_svm))
   #dif_neg = abs(labels_neg -  res_neg_svm)
   
  # min_dist = np.min(res_neg_svm)
  # max_dist = np.max(res_pos_svm)
   thresholds = np.arange(-1,+1,0.0001)

   precision_values = []
   recall_values = []
   for th in thresholds:
       
       tp = len((np.where(res_pos_svm > th)[0]))# True Positive
       fn =len(res_pos_svm) -tp  # False Negative
       tn = len((np.where(res_neg_svm < th))[0]) # True Negative
       fp =len(res_neg_svm)-tn# False Positive

       precision = tp/(fp + tp)
       precision_values.append(precision)
       recall = tp/(fn + tp)
       recall_values.append(recall)
       
   fig, ax = plt.subplots()
  #ax.scatter(recall_values, precision_values)
  #for i  in range(len(thresholds)):
  #     if (i%100==0): ax.annotate("{:.2f}".format(thresholds[i]), (recall_values[i], precision_values[i]))
   ax.plot(recall_values, precision_values)
   #ax.set(xlabel='Recall', ylabel='Precision', title=title+"- Threshold from "+ "{:.2f}".format(min_dist)+"  to  "+"{:.2f}".format(max_dist)+ " ,step 0.001")
   ax.set(xlabel='Recall', ylabel='Precision', title=title+"- Threshold from -1 to 1 ,step 0.0001")

   ax.grid()

   fig.savefig("../"+title+"_RecallPrecision.png")
   plt.show()
       

 # Instead of reading the Hog features and labels from a file, it receives the data from  the Task 2
 # The features and labels are saved anyway in the files: '../train_hog_descs.dat' and   '../labels_train.dat'
def task3(pos_HoGs,neg_HoGs): 
    print('Task 3 - Train SVM and predict confidence values')
    #TODO Create 3 SVMs with different C values, train them with the training data and save them
   
    # 1- Set up the training data
    # labels array will have size of 4500
    labels_pos = np.ones(len(pos_HoGs))
    labels_neg = -1* np.ones(len(neg_HoGs))
    labels =np.array( np.concatenate((labels_pos,labels_neg)),dtype = np.float32)
    # trainingData array will have size of 4500 rows and 3780 columns.
    pos_HoGs_np = np.array(pos_HoGs, dtype=np.float32)
    pos_HoGs_np = np.reshape(pos_HoGs_np, (len(pos_HoGs),3780))
    neg_HoGs_np = np.array(neg_HoGs, dtype=np.float32)
    neg_HoGs_np = np.reshape(neg_HoGs_np, (len(neg_HoGs),3780))
    trainingData =  np.vstack([pos_HoGs_np,neg_HoGs_np])
    
    print('Labels shape:',labels.shape)
    print('Training Data matrix shape:',trainingData.shape)
    
    
    # 2- Set up SVM's parameters
   
    # C-Support Vector Classification.
    #   - n-class classification (n ≥ 2), 
    #   - allows imperfect separation of classes with penalty multiplier C for outliers.
    
    # For C = 0.01
    svm1 = cv.ml.SVM_create()
    svm1.setType(cv.ml.SVM_EPS_SVR)
    svm1.setC(0.01)
    svm1.setDegree( 3 )
    svm1.setP( 0.1 )
    svm1.setKernel(cv.ml.SVM_LINEAR)
    svm1.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-3))
    svm1.train(trainingData, cv.ml.ROW_SAMPLE, labels)
    # Save the model
    svm1.save("../Model_svm1_c001.xml") # Store it by using OpenCV function
    
    # For C = 1
    svm2 = cv.ml.SVM_create()
    #svm2.setType(cv.ml.SVM_C_SVC) 
    svm2.setType(cv.ml.SVM_EPS_SVR)
    svm2.setC(1)
    svm2.setDegree( 3 )
    svm2.setP( 0.1 )
    svm2.setKernel(cv.ml.SVM_LINEAR)
    svm2.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-3))
    svm2.train(trainingData, cv.ml.ROW_SAMPLE, labels)
    # Save the model
    svm2.save("../Model_svm2_c1.xml") # Store it by using OpenCV function
    
    # For C = 10
    svm3 = cv.ml.SVM_create()
    #svm3.setType(cv.ml.SVM_C_SVC) 
    svm3.setType(cv.ml.SVM_EPS_SVR)
    svm3.setC(10)
    svm3.setDegree( 3 )
    svm3.setP( 0.1 )
    svm3.setKernel(cv.ml.SVM_LINEAR)
    svm3.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-3))
    svm3.train(trainingData, cv.ml.ROW_SAMPLE, labels)
    # Save the model
    svm3.save("../Model_svm3_c10.xml") # Store it by using OpenCV function


    # 3- Create a file containing the confidence scores (distance to the margin) for each training sample.
    # set the RAW_OUTPUT flag in the predict method, to get the "distance to the margin"
    # cv.ml_StatModel.predict(	samples[, results[, flags]]	) ->	retval, results
    # enum Flags { UPDATE_MODEL = 1,RAW_OUTPUT=1,COMPRESSED_INPUT =2,PREPROCESSED_INPUT =4,}
    # RAW_OUTPUT : makes the method return the raw results (the sum), not the class label
    _, cs1 = svm1.predict(trainingData, flags =  1) # cs1  = Confidence Score 1
    _, cs2 = svm2.predict(trainingData, flags =  1) # cs2 = Confidence Score 2
    _, cs3 = svm3.predict(trainingData, flags =  1) # cs3 = Confidence Score 3
    
    confidenceScoreFile1 = '../confidencescore_svm_c1e-2.dat' 
    csfile1 = open(confidenceScoreFile1, "w")
    for confidenceScore in cs1:
        print(np.array2string(confidenceScore), file=csfile1)
    csfile1.close()
    
    confidenceScoreFile2 = '../confidencescore_svm_c1.dat' 
    csfile2 = open(confidenceScoreFile2, "w")
    for confidenceScore in cs2:
        print(np.array2string(confidenceScore), file=csfile2)
    csfile2.close()
    
    confidenceScoreFile3 = '../confidencescore_svm_c10.dat' 
    csfile3 = open(confidenceScoreFile3, "w")
    for confidenceScore in cs3:
        print(np.array2string(confidenceScore), file=csfile3)
    csfile3.close()
    
       
        
    # 4- Test the 3 different models
    # then use them to classify the test images and save the results
        
    # For the positive test set 
    filelist_testPos = path_test_2 + 'filenamesTestPos.txt'    
    pos_testfile = open(filelist_testPos,'r')

    hog_dscr = cv.HOGDescriptor()
    
    # Arrays to save the results of the positive test images, considering the models svm1,svm2,svm3
    res_pos_svm1 = [] # C = 0.01
    res_pos_svm2 = [] # C = 1
    res_pos_svm3 = [] # C = 10
    
    for i , line in enumerate(pos_testfile):
        image_name  = path_train_2 +'/pos/'+line.rstrip('\n') 
        img = cv.imread(image_name)
        crop = get_pos_crop(img)
        hog = hog_dscr.compute(crop)
        hog = np.reshape(hog,(1,3780))
        # Prediction for SVM with C = 0.01
        _,response1 = svm1.predict(hog)
        res_pos_svm1 = np.append(res_pos_svm1,response1)
        # Prediction for SVM with C = 1
        _,response2 = svm2.predict(hog)
        res_pos_svm2 = np.append(res_pos_svm2,response2)
        # Prediction for SVM with C = 10
        _,response3 = svm3.predict(hog)
        res_pos_svm3 = np.append(res_pos_svm3,response3)

    pos_testfile.close()
     
    
    # For the Negative Test set
    filelist_testNeg = path_test_2 + 'filenamesTestNeg.txt'
    neg_testfile = open(filelist_testNeg,'r')
    
    # Arrays to save the results of the negative test images, considering the models svm1,svm2,svm3
    res_neg_svm1 = [] # C = 0.01
    res_neg_svm2 = [] # C = 1
    res_neg_svm3 = [] # C = 10
    for i , line in enumerate(neg_testfile):
        # remove end of line char
        image_name  = path_train_2 +'/neg/'+line.rstrip('\n') 
        img = cv.imread(image_name)
        crops = get_neg_crops(img)
        for c in crops:
            hog = hog_dscr.compute(c)
            hog = np.reshape(hog,(1,3780))
            # Prediction for SVM with C = 0.01
            _,response1 = svm1.predict(hog)
            res_neg_svm1 = np.append(res_neg_svm1,response1)
            # Prediction for SVM with C = 1
            _,response2 = svm2.predict(hog)
            res_neg_svm2 = np.append(res_neg_svm2,response2)
            # Prediction for SVM with C = 10
            _,response3 = svm3.predict(hog)
            res_neg_svm3 = np.append(res_neg_svm3,response3)
    
    neg_testfile.close()       

    # Task 4
    plot_results( res_pos_svm1,res_neg_svm1, "C = 0.01")
    plot_results( res_pos_svm2,res_neg_svm2, "C = 1")
    plot_results( res_pos_svm3,res_neg_svm3, "C = 10")
    return(svm1, cs1, res_pos_svm1,res_neg_svm1,svm2, cs2,res_pos_svm2,res_neg_svm2,svm3,cs3,res_pos_svm3,res_neg_svm3)
        
        



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
    pos_HoGs,neg_HoGs = task2()

    # Task 3 - Train SVM 
    # Instead of reading the Hog features and labels from a file, it receives the results from  the Task 3
    # Everything that is returned by the function is saved as a file
    svm1, cs1, res_pos_svm1,res_neg_svm1,svm2, cs2,res_pos_svm2,res_neg_svm2,svm3,cs3,res_pos_svm3,res_neg_svm3 = task3(pos_HoGs,neg_HoGs)

    # Task 5 - Multiple Detections
    #task5()

