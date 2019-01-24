import numpy as np
from ImageDatasetLoader import ImageDatasetLoader

class SegmentationDatasetLoader(ImageDatasetLoader):

    def __init__(self, input_path, label_path, color_map):

        super().__init__(input_path, label_path)

        # Check for unusualities in the given directory
        check()

        self.color_map = color_map


    def create_masks(self):
        mask = []
        for i in color_map:
                color_img = []
                for j in range(3):
                        color_img.append(np.ones((img.shape[:-1])) * i[j]) 
    
                img2 = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)
    
        mask.append(np.uint8((img == img2).sum(axis = -1) == 3))
      return np.array(mask)

