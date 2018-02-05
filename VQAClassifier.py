import keras
import numpy as np

class VQAClassifier:
    def __init__(self):
        pass        

    def get_image_indices_for_questions(self,question_image_ids,image_ids):
        image_indices_for_questions = []

        for i in range(len(question_image_ids)):
            image_indices_for_questions.append(image_ids.index(question_image_ids[i]))

        return image_indices_for_questions

    def build_vqa_dataset(self,images_tensor,questions_tensor,answers,image_indices_for_questions,vqa_classes):
        X_images = images_tensor[image_indices_for_questions]
        X_questions = questions_tensor

        X = np.concatenate([X_images,X_questions],axis=1)
        Y = []

        for ans in answers:
            if ans in vqa_classes:
                Y.append(vqa_classes.index(ans))
            else:
                Y.append(vqa_classes.index("-NA-"))

        Y_one_hot = keras.utils.to_categorical(Y,num_classes=len(vqa_classes))

        return X,Y,Y_one_hot

    def build_vqa_model(self,num_inputs,num_units_for_layers,num_classes,activation):
        model = keras.models.Sequential()

        if len(num_units_for_layers) != 0:
            model.add(keras.layers.Dense(num_units_for_layers[0],activation=activation,input_shape=[num_inputs]))

        for i in range(1,len(num_units_for_layers)):
            model.add(keras.layers.Dense(num_units_for_layers[i],activation=activation))

        if len(num_units_for_layers) == 0:
            model.add(keras.layers.Dense(num_classes,activation='softmax',input_shape=[num_inputs]))
        else:
            model.add(keras.layers.Dense(num_classes,activation='softmax'))

        sgd = keras.optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.80)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=["accuracy"])
        return model