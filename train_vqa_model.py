from VQAClassifier import VQAClassifier
import numpy as np
import json

model_name = "model_X"

dim_img_vector = 4096
dim_questions_vector = 300

vqa_classifier = VQAClassifier()
X = np.load("X.npy")
Y_one_hot = np.load("Y_one_hot.npy")
print("Loaded data...")
model = vqa_classifier.build_vqa_model(dim_questions_vector+dim_img_vector,[512,512],Y_one_hot.shape[1],activation='tanh')
model.summary()
print("Built model..")
print("Commencing training...")
history = model.fit(X,Y_one_hot,epochs=10,validation_split=0.2)
print(type(history.history))
model.save(model_name + ".h5")
print("Training completed...")
with open(model_name + "_metrics.json","w") as fp:
	json.dump(history.history,fp)
print("Results saved.")