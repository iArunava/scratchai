import numpy as np
from ImageDatasetLoader import ImageDatasetLoader

class SegmentationDatasetLoader(ImageDatasetLoader):

    def __init__(self, input_path, label_path, color_map):
        '''
        Constructor for the Segmentation Dataset Loader

        Arguemnts:
        - input_path : 
        - label_path : 
        - color_map - dict - a dictionary where the key is the class name
                             and the value is a tuple or list with 3 elements
                             one for each channel. So each key is a RGB value.
        '''

        super().__init__(input_path, label_path)

        # Check for unusualities in the given directory
        self.check()

        self.color_map = color_map
        self.classes = self.color_map.keys()
        self.colors = self.color_map.values()

    def create_masks(self, image):
        '''
        A class that creates masks for each of the classes

        Arguments:
        - Image.PIL - Semantic Segmented Image where each pixel is colored
                      with a specific color
                      The Image is of size H x W x C
                      where H is the height of the image
                            W is the width of the image
                            C is the number of channels (3)

        Returns:
        - np.ndarray - of size N x H x W
                       where N is the number of classes
                             H is the height of the image
                             W is the width of the image
        '''
        
        if image.shape[-1] > 3 or image.shape[-1] < 3:
            raise RuntimeError('The image passed has more than expected channels!')
        
        masks = []
        for ii in self.color_map:
            color_img = []
            for j in range(3):
                color_img.append(np.ones((img.shape[:-1])) * ii[j])
            img2 = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)
            masks.append(np.uint8((image == img2).sum(axis=-1) == 3))

        return np.array(masks)


    def decode_segmap(self, image):
        '''
        The method helps one get a colorful image where each color corresponds to each class

        Arguments:
        - Image - np.array - A 2D Image where each pixel position is a number indicating
                             the class to which is belongs
        
        Returns:
        - np.array - H x W x C
                    where each pixel position [x, y, :]
                    is a color representing its RGB color which is passed in
                    with the color_map while initializing this class
        '''

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for label in range(0, self.num_classes):
            r[image == label] = self.colors[label][0]
            g[image == label] = self.colors[label][1]
            b[image == label] = self.colors[label][2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb
