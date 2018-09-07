import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout, Conv2DTranspose, Concatenate, Cropping2D
import numpy as np
import cv2




def process_seg_map(seg_map):
    """
    input: Segmentation map
           dimensions : (d,d,2)
           
    returns: Image of dimensions: (d,d,3)
             formed after applying threshold function to seg_map
    """
    output_image = np.ones((seg_map.shape[0], seg_map.shape[1], 3)) *  255 # initialize with white all
    max_map = np.argmax(seg_map, axis=2)
    for i in range(max_map.shape[0]):
        for j in range(max_map.shape[1]):
            if max_map[i][j] == 1:
                output_image[i][j] = np.array([255,0,0]) # red
    assert output_image.shape == (seg_map.shape[0], seg_map.shape[1], 3)
    return output_image
    
    




def make_model():
    # first block of contracting path
    x_input = Input(shape=(572,572,1))
    x = Convolution2D(64, (3,3))(x_input)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64, (3,3))(x)
    x_first = Activation("relu")(x)
    x = BatchNormalization()(x_first)
    x = MaxPooling2D()(x)  # reduce the size
    # second block of contracting path
    x = Convolution2D(128, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, (3,3))(x)
    x_second = Activation("relu")(x)
    x = BatchNormalization()(x_second)
    x = MaxPooling2D()(x)
    # third block of contracting path
    x = Convolution2D(256, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, (3,3))(x)
    x_third = Activation("relu")(x)
    x = BatchNormalization()(x_third)
    x = MaxPooling2D()(x)
    # fourth block of contracting path
    x = Convolution2D(512, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, (3,3))(x)
    x_fourth = Activation("relu")(x)
    x = BatchNormalization()(x_fourth)
    x = MaxPooling2D()(x)
    
    # now the bottleneck layer, composed of two conv layers
    x = Convolution2D(1024, (3,3))(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)
    x = Convolution2D(1024, (3,3))(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)
    
    # now we begin with the decoding
    # decoder first block
    x = Conv2DTranspose(512, (3,3), strides=(2,2))(x)
    """concatenate the previous corresponding encoder layer in next lines"""
    x_fourth = Cropping2D()(x_fourth)
    x = Concatenate([x_fourth, x], axis=2)
    x = Convolution2D(512, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    # decoder second block
    x = Conv2DTranspose(256, (3,3), strides=(2,2))(x)
    """concatenate the previous corresponding encoder layer in next lines"""
    x_third = Cropping2D()(x_third)
    x = Concatenate([x_third, x], axis=2)
    x = Convolution2D(256, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    #decoder third block
    x = Conv2DTranspose(128, (3,3), strides=(2,2))(x)
    """concatenate the previous corresponding encoder layer in next lines"""
    x_second = Cropping2D()(x_second)
    x = Concatenate([x_second, x], axis=2)
    x = Convolution2D(128, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    # decoder fourth block
    x = Conv2DTranspose(64, (3,3), strides=(2,2))(x)
    """concatenate the previous corresponding encoder layer in next lines"""
    x_first = Cropping2D()(x_first)
    x = Concatenate([x_first, x], axis=2)
    x = Convolution2D(64, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64, (3,3))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    # now the segmentation map, (d,d,2) in dimensions
    output = Convolution2D(2,(1,1))(x)   # the final output
    output = Activation("softmax")(output)
    
    # now make the model
    model = Model(inputs=x_input, outputs=output)
    return model

    
model = make_model()
model.compile(loss="categorical_crossentropy",optimizer="adam")
    
    
