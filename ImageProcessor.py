import cv2
import keras
import numpy as np
from imagenet_classes import class_names
class ImageProcessor:
    def __init__(self):
        pass
    def build_vgg_16_model(self,vgg_weights_path):
        model = keras.models.Sequential()
        ##Block 1 layers
        model.add(keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1',input_shape=[224,224,3]))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        
        ##Block 2 Layers
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        
        #Block 3 Layers
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        
        #Block 4 Layers
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        
        #Block 5 Layers
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        
        model.add(keras.layers.Flatten(name='flatten'))
        
        #Fully connected block
        model.add(keras.layers.Dense(4096, activation='relu', name='fc1'))
        ##IMP: THE FOLLOWING LAYER IS THE INTERMEDIATE LAYER OF INTEREST
        ## IMAGE FEATURE VECTOR IS USED AS THE OUTPUT OF THIS LAYER
        model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
        
        model.add(keras.layers.Dense(1000, activation='softmax', name='predictions'))
        model.load_weights(vgg_weights_path)
        return model

    def get_top_ten_predictions(self,images,model):
        prediction_labels_probs = np.zeros([len(images),10,2])
        predictions = model.predict(images)
        top_ten_preds = np.argsort(predictions)[:,-10:]
        top_ten_probs = np.sort(predictions)[:,-10:]
        i = 0
        for preds,probs in zip(top_ten_preds,top_ten_probs):
            pred_labels = class_names[preds]
            prediction_labels_probs[i] = np.concatenate([pred_labels,probs])
            i = i+1
        return prediction_labels_probs

    def get_image_feature_vectors(self,images,model):
        intermediate_layer_model = keras.Model(inputs=model.input,outputs=model.get_layer('fc2').output)
        image_feature_vectors = intermediate_layer_model.predict(images.reshape(len(images),224,224,3))
        return np.squeeze(image_feature_vectors)