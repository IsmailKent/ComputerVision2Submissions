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
        patches = []
        labels = []
        train = self.train_images_list
        gt = self.gt_segmentation_maps_list
        classes = self.class_colors
        patch_size = self.patch_size
        # loop over images
        for i in range(len(train)):
            image = train[i]
            segmentation = gt[i]
            for c in classes:
                # get binary matrix of entries with this class
                binary_matrix = segmentation.copy()
                binary_matrix = binary_matrix == c
                # get x,y lower right corner of patch size square
                positions = self.get_positions_of_patch(binary_matrix, patch_size)
                for pos in positions:
                    x , y = pos
                    image_patch = image[x-patch_size:x, y-patch_size:y]
                    """
                    # sanity check, all cells in this patch have right class
                    seg_patch = segmentation[x-patch_size:x, y-patch_size:y]
                    print( np.sum(seg_patch==c) == patch_size*patch_size*3)
                    """
                    patches.append(image_patch)
                    labels.append(c)
        np.random.shuffle(patches)            
        return  patches , labels

    # function using dynamic programming to get the first square 
    #of 1s of size patch_size given a binary matrix
    # return rights lower corner
    def get_positions_of_patch(self,matrix,patch_size):
        x,y , _ = matrix.shape
        dp = np.zeros((x+1,y+1))
        positions = []
        for i in range(1,x+1):
            for j in range(1,y+1):
                if (matrix[i-1][j-1][0]):
                    dp[i][j] = min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1
                    if (dp[i][j]>= patch_size):
                        positions.append( ( i,j ))
                
     # not found
        return positions



    # feel free to add any helper functions


