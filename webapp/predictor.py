import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from urllib.request import urlopen

class Predictor:
    def __init__(self):
        # predefined constant values
        self.MODEL_URL = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2'
        self.IMAGE_SIZE = 384
        self.LABEL_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'

        # downloading labels from the url
        file = urlopen(self.LABEL_URL)
        self.labels = []
        for line in file:
            self.labels.append(line.decode().strip()) # decode from bytes to str

        # loading the model
        self.model = hub.load(self.MODEL_URL)
        self.warmup()

    def warmup(self):
        """Proived a warmup input for the model when the model is loaded
        Args:
            None
        Returns:
            None
        """
        inp_shape = [1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3]
        inp = tf.random.uniform(inp_shape, 0, 1)
        self.model(inp)

    def adjust_images(self, img, size = None):
        """Function to adjust images
        Args:
            img: Object - image to be adjusted
            size: Integer - (Default: None) size of the image to be adjusted
        Returns:
            Object - adjusted image
        """
        img = np.array(img)
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        
        img = img.astype(np.float32)
        img = img / 255.0 
        
        if not size:
            size = self.IMAGE_SIZE
        img = tf.image.resize_with_pad(img, size, size)

        return img

    def predict(self, img):
        """Function to predict the image classes
        Args:
            img: Object - image to be classified
        Returns:
            Dictionary - predicted labels and probabilities
        """
        adjusted_img = self.adjust_images(img)
        prob = tf.nn.softmax(self.model(adjusted_img)).numpy()
        top_5 = np.argsort(prob)[0][::-1][:5]

        pred = {}
        for i in top_5:
            pred[self.labels[i]] = prob[0][i]
        
        return pred