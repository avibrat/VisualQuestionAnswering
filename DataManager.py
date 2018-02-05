import cv2
import numpy as np
import json
import os
class DataManager:
    def __init__(self):
        pass
    def load_images_from_directory(self,directory):
        images = []
        image_ids = []
        for file in os.listdir(directory):
            image = cv2.imread(os.path.join(directory,file))
            image = cv2.resize(image,(224,224))
            images.append(image)
            image_ids.append(int(file[13:25]))
        return np.asarray(images), image_ids

    def load_questions_for_images(self,image_ids,questions_json,annotations_json):
        questions = json.load(open(questions_json))["questions"]
        annotations = json.load(open(annotations_json))["annotations"]
        questions_list = []
        answers_list = []
        question_image_ids_list = []
        for qn,an in zip(questions,annotations):
            if qn["image_id"] in image_ids:
                questions_list.append(qn["question"])
                answers_list.append(an["multiple_choice_answer"])
                question_image_ids_list.append(qn["image_id"])
        return (questions_list,answers_list,question_image_ids_list)