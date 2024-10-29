import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.epoch = 0
        self.index = 0

        self._load_data()

    def _load_data(self):
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        self.images = []
        self.image_labels = []
        for file_name in os.listdir(self.file_path):
            if file_name.endswith('.npy'):
                image = np.load(os.path.join(self.file_path, file_name))
                image_resized = resize(image, self.image_size)
                self.images.append(image_resized)
                self.image_labels.append(self.labels[file_name.split('.')[0]])

        self.images = np.array(self.images)
        self.image_labels = np.array(self.image_labels)

        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        indices = np.arange(len(self.images))
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.image_labels = self.image_labels[indices]

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        start = self.index
        end = start + self.batch_size

        if end > len(self.images):
            batch_images = np.concatenate((self.images[start:], self.images[:(end - len(self.images))]), axis=0)
            batch_labels = np.concatenate((self.image_labels[start:], self.image_labels[:(end - len(self.image_labels))]), axis=0)
            self.index = (end - len(self.images))
            self.epoch += 1
            if self.shuffle:
                self._shuffle_data()
        elif end == len(self.images):
            batch_images = self.images[start:end]
            batch_labels = self.image_labels[start:end]
            self.index = end
            if self.shuffle:
                self._shuffle_data()
        else:
            batch_images = self.images[start:end]
            batch_labels = self.image_labels[start:end]
            self.index = end

        batch_images = np.array([self.augment(img) for img in batch_images])

        return batch_images, batch_labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring and np.random.rand() > 0.5:
            img = np.fliplr(img)
        if self.rotation:
            rotations = np.random.choice([0, 1, 2, 3])
            img = np.rot90(img, rotations)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        return class_dict.get(x, "Unknown")

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        plt.figure(figsize=(10, 10))
        for i in range(self.batch_size):
            plt.subplot(1, self.batch_size, i + 1)
            plt.imshow(images[i])
            plt.title(self.class_name(labels[i]))
            plt.axis('off')
        plt.show()

