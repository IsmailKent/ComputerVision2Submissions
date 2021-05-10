import numpy as np


class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, classes_colors, patch_size):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = classes_colors
        self.patch_size = patch_size

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        class_0_list = []
        class_1_list = []
        class_2_list = []
        class_3_list = []
        train = self.train_images_list
        gt = self.gt_segmentation_maps_list
        classes = self.class_colors
        patch_size = self.patch_size
        # loop over images
        for i in range(train.shape[0]):
            image = train_images_list[i]
            segmentation = gt_segmentation_maps_list[i]
            for c in classes:
                # get binary matrix of entries with this class
                binary_matrix = segmentation.copy()
                binary_matrix = binary_matrix[binary_matrix == c]
                # get x,y lower right corner of patch size square
                x,y = get_position_of_patch(segmentation, patch_size)
                patch = image[x-patch_size:x, y-patch_size:y]
                if (c == 0):
                    class_0_list.append(patch)
                elif (c==1):
                    class_1_list.append(patch)
                elif (c==2):
                    class_2_list.append(patch)
                elif (c==3):
                    class_3_list.append(patch)
        return class_0_list , class_1_list , class_2_list , class_3_list

    # function using dynamic programming to get the first square 
    #of 1s of size patch_size given a binary matrix
    # return rights lower corner
    def get_position_of_patch(matrix,patch_size):
        x,y = matrix.shape
        dp = np.zeros((x+1,y+1))
        for i in range(1,x+1):
            for j in range(1,y+1):
                if (matrix[i-1][j-1] == 1):
                    dp[i][j] = min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1
                    if (dp[i][j]== patch_size):
                        return i,j
                
     # not found
        return -1,-1



    # feel free to add any helper functions


