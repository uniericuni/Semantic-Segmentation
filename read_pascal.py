from scipy import misc
from config import *
import numpy as np

class PascalReader:
    
    def __init__(self):
        self.trainlist_filename = './VOC2011/ImageSets/Segmentation/train.txt'
        self.testlist_filename = './VOC2011/ImageSets/Segmentation/test.txt'
        self.train_image_dir = './VOC2011/JPEGImages/'
        self.train_label_dir = './VOC2011/SegmentationClass/'
        train_list = open(self.trainlist_filename)
        self.train_names = train_list.readlines()
        self.train_names = [x.strip() for x in self.train_names]
        self.current_image = 0;
        self.patch_length = 224;
        self.padding_length = INIT_PADDING

    def next_batch(self, batch_size=BATCH_SIZE):
        imageName = self.train_image_dir + self.train_names[self.current_image] + '.jpg'
        labelName = self.train_label_dir + self.train_names[self.current_image] + '.png'
        image = misc.imread(imageName)
        label = misc.imread(labelName)
        HEIGHT = image.shape[0]
        WIDTH = image.shape[1]
        tmpImage = np.zeros((1, 2*self.padding_length+image.shape[0], 2*self.padding_length+image.shape[1], image.shape[2]))
        tmpImage[0,self.padding_length:self.padding_length+image.shape[0],self.padding_length:self.padding_length+image.shape[1],:] = image[:,:,:]
        tmpLabel = np.zeros((1, label.shape[0], label.shape[1], label.shape[2]))
        tmpLabel[0,:,:,:] = label[:,:,:]

        self.current_image += 1

        return tmpImage.tolist(), tmpLabel.tolist()

        # for x in np.linspace(0, image.shape[0]-self.padding_length-1 , 4, dtype = np.int16):
        #     for y in np.linspace(0, image.shape[1]-self.padding_length-1 , 5, dtype = np.int16):
        #         tmpImage = np.zeros((1, 2*self.padding_length+self.patch_length, 2*self.padding_length+self.patch_length, 3))
        #         tmpImage[0,self.padding_length,self.padding_length] = image[x:x+elf.patch_length,p:p+elf.patch_length,:]
        #         tmpLabel = np.zeros((1, self.patch_length, self.patch_length, 3))
        #         tmpLabel[0,:,:,:] = label[x:x+elf.patch_length,p:p+elf.patch_length,:]
    
    # TODO
    def next_test(self):
        imageName = self.train_image_dir + self.train_names[self.current_image] + '.jpg'
        labelName = self.train_label_dir + self.train_names[self.current_image] + '.png'
        image = misc.imread(imageName)
        label = misc.imread(labelName)
        tmpImage = np.zeros((1, 2*self.padding_length+image.shape[0], 2*self.padding_length+image.shape[1], image.shape[2]))
        tmpImage[0,self.padding_length:self.padding_length+image.shape[0],self.padding_length:self.padding_length+image.shape[1],:] = image[:,:,:]
        tmpLabel = np.zeros((1, label.shape[0], label.shape[1], label.shape[2]))
        tmpLabel[0,:,:,:] = label[:,:,:]

        self.current_image += 1

        return tmpImage.tolist(), tmpLabel.tolist()
