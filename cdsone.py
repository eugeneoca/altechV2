# Author: AL-TECHNOLOGIES

from __future__ import print_function
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# GUI 
from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import (Qt, pyqtSignal, QThread, pyqtSlot)
from altech_gui import *
import form_RegisterIdentity as frm_register
import form_TrainIdentity as frm_train
import form_EraseIdentity as frm_erase

import os
import sys
import socket
import threading
import time
import cv2 as cv
import pickle
from sklearn.externals import joblib
from mtcnn.mtcnn import MTCNN
from engine import facenet
import vlc
import numpy as np
import tensorflow as tf
import os

# Custom Library
import lib

# Settings
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Camera(QThread):
	index = ""
	mode = False
	stop_flag = False
	win = ""
	detector = MTCNN()
	result_callback = pyqtSignal(object, object, int)
	
	def run(self):
		# Reduce size by selecting randomly
		
		font = cv.FONT_HERSHEY_SIMPLEX
		labels_name, labels, hyperspheres = joblib.load("model_encodes.ds")
		sample_size = 20 # x/200
		hyperspheres_reduced = []
		labels_reduced = []
		
		for i, identity in enumerate(labels_name):
			sample_space = np.arange(len(hyperspheres[i]))
			np.random.shuffle(sample_space)
			random_selection = sample_space[:sample_size]
			for key in random_selection:
				hyperspheres_reduced.append(hyperspheres[i][key])
				labels_reduced.append(labels[i][key])
		if self.mode == True:
			inst = vlc.Instance("--verbose=-1 --file-caching=0 --network-caching=0 --vout=dummy --no-snapshot-preview --no-osd --preferred-resolution=-1 --transform-type=hflip")
			newCamera = inst.media_player_new()
			media = inst.media_new("rtsp://admin:1234abcd@192.168.1.64:554/PSIA/streaming/channels/"+str(self.index+1)+"02")
			newCamera.set_media(media)
			newCamera.set_hwnd(self.win)
			newCamera.play()
			frame = None
			imgname = "frames\\1.jpg"
			facenet.load_model("models/20180402-114759/20180402-114759.pb")
			with tf.Session().as_default() as sess:
				images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
				embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
				phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
				while not self.stop_flag:
					try:
						if(newCamera.video_take_snapshot(0, imgname, 0, 0)) == 0:
							frame = cv.imread(imgname)
								
						faces = self.detector.detect_faces(frame)
						for face in faces:
							bounding_box = face['box']
							keypoints = face['keypoints']
							face_patch = frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], :]
							face_patch = cv.resize(face_patch, (150,150))
							face_patch = facenet.prewhiten(face_patch)
							face_patch = np.stack(face_patch)
							cv.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
							feed_dict = {images_placeholder: [face_patch], phase_train_placeholder: False}
							embs = sess.run(embeddings, feed_dict=feed_dict)
							dataset = np.array(hyperspheres_reduced)
							result = facenet.distance(dataset, embs, 0)<=1.2
							identity = self.tally(result, labels_reduced, 0.5, labels_name)
							self.result_callback.emit(frame, identity, self.index)
					except Exception as error:
						pass
		else:
			cap = cv.VideoCapture(self.index)
			while not cap.isOpened():
				pass
			facenet.load_model("models/20180402-114759/20180402-114759.pb")
			with tf.Session().as_default() as sess:
				images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
				embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
				phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
				while not self.stop_flag:
					ret, frame = cap.read()
					try:
						frame = cv.flip(frame, 1)
						faces = self.detector.detect_faces(frame)
						for face in faces:
							bounding_box = face['box']
							keypoints = face['keypoints']
							face_patch = frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], :]
							face_patch = cv.resize(face_patch, (150,150))
							face_patch = facenet.prewhiten(face_patch)
							face_patch = np.stack(face_patch)
							cv.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
							feed_dict = {images_placeholder: [face_patch], phase_train_placeholder: False}
							embs = sess.run(embeddings, feed_dict=feed_dict)
							dataset = np.array(hyperspheres_reduced)
							result = facenet.distance(dataset, embs, 0)<=1.2
							identity = self.tally(result, labels_reduced, 0.5, labels_name)
							self.result_callback.emit(frame, identity, self.index)
					except Exception as error:
						print(error)
					
		print("Camera Index("+str(self.index)+") terminated.")
		
	def tally(self, result, labels_reduced, target_rate, labels_name):
		bin_result = result*1
		key_result = bin_result.nonzero()[0]
		#print(key_result)
		voted_names = np.take(labels_reduced, key_result)
		votes = np.bincount(voted_names)
		total = len(labels_reduced)/len(labels_name)
		scores = votes/total
		winner = (scores>target_rate)*1
		is_unique = len(np.nonzero(winner)[0])==1
		identity = "unknown"
		if is_unique:
			identity = np.argmax(scores)
			identity = labels_name[identity]
		else:
			identity = "unknown"
			
		return identity
			
	def set_index(self, key):
		self.index = key
		
	def set_mode(self, mode):
		self.mode = mode
		
	def set_hwnd(self, handle):
		self.win = handle
		
	def stop(self):
		self.stop_flag = True
		
# Application
class Main(QMainWindow, Ui_MainWindow):

	mode = False
	camera_limit = 1
	camera_feeds = []
	camWindow_array = []
	
	def __init__(self):
		super(Main, self).__init__()
		self.setupUi(self)
		
		# Available Camera Slots
		self.camWindow_array.append(self.camWindow1)
		self.camWindow_array.append(self.camWindow2)
		self.camWindow_array.append(self.camWindow3)
		self.camWindow_array.append(self.camWindow4)
		
		self.camera_limit = len(self.camWindow_array)
		
		rtsp_mode = QMessageBox.question(self, 'AL-TECHNOLOGIES', 'Do you want to use RTSP Feed?', QMessageBox.Yes, QMessageBox.No)
		if rtsp_mode == QMessageBox.Yes:
			print("Connecting RTSP Feed...")
			self.mode = True
			for i in range(self.camera_limit):
				cam = Camera(self)
				cam.set_index(i)
				cam.set_mode(self.mode)
				cam.set_hwnd(self.camWindow_array[i].winId())
				cam.result_callback.connect(self.onNewResult)
				cam.start()
				self.camera_feeds.append(cam)
		else:
			print("Connecting Local Feeds...")
			self.mode = False
			for i in range(self.camera_limit):
				cam = Camera(self)
				cam.set_index(i)
				cam.set_mode(self.mode)
				cam.result_callback.connect(self.onNewResult)
				cam.start()
				self.camera_feeds.append(cam)
			
	@pyqtSlot(object, object, int)
	def onNewResult(self, frame, identity, index):
		rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
		pic = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
		new_picture = QPixmap.fromImage(pic)
		self.camWindow_array[index].setPixmap(new_picture)
		self.txt_log.append(identity+ " detected on Camera Index("+str(index)+")")
		self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())
			
	def closeEvent(self, event):
	
		# Camera feeds termination
		for cam in self.camera_feeds:
			cam.stop()
			
		time.sleep(5)

app = QtWidgets.QApplication([])
application = Main()
application.show()
sys.exit(app.exec())