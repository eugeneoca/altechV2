import tensorflow as tf
from engine import facenet
import os
import numpy as np
from sklearn.externals import joblib
import face_recognition as recognizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

facenet.load_model("models/20180402-114759/20180402-114759.pb")

directory = "datasets"
datasets = facenet.get_dataset(directory)

with tf.Session().as_default() as sess:
	images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
	embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
	phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
	
	hyperspheres = []
	labels = []
	labels_name = []
	
	for label, dataset in enumerate(datasets):
		hyperspheres_local = []
		labels_local = []
		print("Training... "+ dataset.__str__())
		images = facenet.load_data(dataset.image_paths, do_random_crop=True, do_random_flip=True, image_size=150, do_prewhiten=True)
		
		feed_dict = {images_placeholder: images, phase_train_placeholder: False}
		embs = sess.run(embeddings, feed_dict=feed_dict)
		for e in embs:
			hyperspheres_local.append(e)
			labels_local.append(label)
			
		hyperspheres.append(hyperspheres_local)
		labels.append(labels_local)
		labels_name.append(dataset.name)
		
	try:
		filename = "model_encodes.ds"
		joblib.dump([labels_name, labels, hyperspheres], filename)
		print("Model has been saved as "+filename)
	except Exception as error:
		print(error)
		
	print("Training finished.")

