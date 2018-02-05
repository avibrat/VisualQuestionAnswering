'''Script to precompute VGG 4096-features for images
	Ran it several times,changing the dataset each time,; to extract and concatenate features in batches to get the whole
	feature vector for the ~1000 images
'''
from DataManager import DataManager
from ImageProcessor import ImageProcessor
import numpy as np
feats_so_far = np.load("image_features_vgg_1000.npy")
print(feats_so_far.shape)
data_manager = DataManager()
images,img_ids = data_manager.load_images_from_directory("C:/Users/avi27/Desktop/Datasets/VQA/val/")
print("Loaded images...")
image_processor = ImageProcessor()
vgg16_model = image_processor.build_vgg_16_model("vgg16_weights_tf_dim_ordering_tf_kernels.h5")
print("Built model....")
img_feat_vecs = image_processor.get_image_feature_vectors(images,vgg16_model)
print("Obtained image feature vectors")
np.save("image_features_vgg_1000",np.concatenate([feats_so_far,img_feat_vecs],axis=0))
print("Saved feature vectors....Done")