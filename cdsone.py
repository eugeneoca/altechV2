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
from PyQt5.QtCore import (Qt, pyqtSignal, QThread, pyqtSlot, QCoreApplication)
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
from threading import Thread,Lock
from serial import *
import datetime
import subprocess
import shutil

# Google Drive API
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file as File, client, tools
from apiclient.http import MediaFileUpload

# Custom Library
import lib

# Settings
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GSM(QThread):
	"""
	GSM Module Object
	"""
	
	port = ""
	baud_rate = 0
	serial_connection = ""
	
	def run(self):
		print("Trying to connect GSM...")
		print("[GSM PORT] Connecting to "+self.port)
		print("[GSM PORT] Baud rate "+str(self.baud_rate))
		self.connect()
		print("[GSM PORT] Checking connectivity...")
		try:
			self.serial_connection.write(b'AT\r')
		except:
			print("GSM Device not found.")
		
		
	def set_port(self, port):
		self.port = port
		
	def set_baud_rate(self, baud_rate):
		self.baud_rate = baud_rate
		
	def connect(self):
		try:
			self.serial_connection = Serial(self.port, self.baud_rate)
		except:
			return False
		return True
		
	def sendMessage(self, number, message):
		self.serial_connection.write(b'ATZ\r')
		sleep(0.5)
		self.serial_connection.write(b'AT+CMGF=1\r')
		sleep(0.5)
		self.serial_connection.write(b'''AT+CMGS="''' + number.encode() + b'''"\r''')
		sleep(0.5)
		self.serial_connection.write(message.encode() + b"\r")
		sleep(0.5)
		self.serial_connection.write(bytes([26]))
		sleep(0.5)
		
	def get_status(self):
		return self.serial_connection.isOpen()
		
	def disconnect(self):
		self.serial_connection.close()

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
		sample_size = 40 # x/200
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
						
		try:
			cap.release()
		except:
			pass
		try:
			newCamera.release()
		except:
			pass
					
		print("Camera Index("+str(self.index)+") terminated.")
		
	def tally(self, result, labels_reduced, target_rate, labels_name):
		bin_result = result*1
		key_result = bin_result.nonzero()[0]
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
		
class UploadElement():
	
	path = ""
	
	def __init__(self, path):
		self.path=path
		
	def get_path(self):
		return  self.path
		
# Application
class Main(QMainWindow, Ui_MainWindow):

	mode = False
	camera_limit = 1
	camera_feeds = []
	camWindow_array = []
	lock = Lock()
	gsm = ""
	indicator = ""
	upload_processes = []
	save_local = False
	upload_manager = None
	stop_upload = False
	
	def __init__(self):
		super(Main, self).__init__()
		self.setupUi(self)
		
		try:
			self.gsm = GSM(self)
			self.gsm.set_port("COM7")
			self.gsm.set_baud_rate(9600)
			self.gsm.connect()
			self.gsm.start()
		except Exception as error:
			print(error)
		try:
			print("[INDICATOR] Connecting indicators...")
			self.indicator = Serial("COM3", 9600)
		except Exception as error:
			print(error)
		
		# Available Camera Slots
		self.camWindow_array.append(self.camWindow1)
		self.camWindow_array.append(self.camWindow2)
		self.camWindow_array.append(self.camWindow3)
		self.camWindow_array.append(self.camWindow4)
		
		self.btn_RegisterIdentity.clicked.connect(self.show_RegistrationForm)
		self.btn_EraseIdentity.clicked.connect(self.show_EraseIdentityForm)
		
		self.upload_manager = threading.Thread(target=self.uploadManager)
		self.upload_manager.daemon = True
		self.upload_manager.start()
		
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
				
		scope = "https://www.googleapis.com/auth/drive"
		storage = "token.json"
		credentials = "credentials.json"
		
		try:
			store = File.Storage(storage)
			creds = store.get()
			if not creds or creds.invalid:
				flow = client.flow_from_clientsecrets(credentials, scope)
				creds = tools.run_flow(flow, store)
			self.service = build('drive', 'v3', http=creds.authorize(Http()))
			self.cloud_directory_service = build('drive', 'v3', http=creds.authorize(Http()))
			self.local_directory_service = build('drive', 'v3', http=creds.authorize(Http()))
			self.root = "1Fr8j9SfgZ7uDvX6i7gHHrjYV7GFq_0mO"
		except Exception as error:
			print(error)
			self.save_local=1
			print("No internet connection.")
			print("Will be saving on local.")
			
	def show_EraseIdentityForm(self):
		self.window = QMainWindow()
		self.form_EraseProperties = frm_erase.Ui_MainWindow()
		self.ui = self.form_EraseProperties.setupUi(self.window)
		identity_folders = os.listdir("datasets")
		for identity in identity_folders:
			self.form_EraseProperties.cmb_IdentityList.addItem(identity)

		self.form_EraseProperties.btn_Erase.clicked.connect(self.eraseIdentity)
		self.window.show()
		
	def eraseIdentity(self):
		self.window.close()
		index = self.form_EraseProperties.cmb_IdentityList.currentIndex()
		name = self.form_EraseProperties.cmb_IdentityList.itemText(index)
		if name=="":
			print("Invalid path.")
			QMessageBox.about(self, "CDS-1", "Invalid Path.")
			return
			
		for feed in self.camera_feeds:
			feed.stop()
			
		time.sleep(3)
		self.window.close()
		self.close()
		self.stop_upload = True
		directory = "datasets\\"+name
		print("Removing identity "+name)
		shutil.rmtree(directory)
		# Retrain
		print("Training our model...")
		cmd_train = subprocess.Popen("python train.py")
		while cmd_train.poll() is None:
			pass
		
		print("Done.")
		QMessageBox.about(self, "CDS-1", directory+" has been deleted.")
		exit(1)
			
	def show_RegistrationForm(self):
		self.window = QMainWindow()
		self.form_registerProperties = frm_register.Ui_MainWindow()
		self.ui = self.form_registerProperties.setupUi(self.window)
		self.form_registerProperties.txt_fname.textChanged.connect(self.updateDirectoryTextBox)
		self.form_registerProperties.txt_lname.textChanged.connect(self.updateDirectoryTextBox)
		self.form_registerProperties.btn_startRegistration.clicked.connect(self.startRegistration)
		self.form_registerProperties.btn_stopRegistration.clicked.connect(self.stopRegistration)
		self.window.show()
		
	def startRegistration(self):
		fname = self.form_registerProperties.txt_fname.text()
		lname = self.form_registerProperties.txt_lname.text()
		if fname=="" or lname=="":
			self.form_registerProperties.txt_process.setText("Failed to start. Please click stop.")
			return
		dataset_name = fname.lower()+'-'+lname.lower()
		for feed in self.camera_feeds:
			feed.stop()
			
		time.sleep(3)
		self.window.close()
		self.close()
		self.stop_upload = True
		try:
			cmd_register = subprocess.Popen("python register.py "+dataset_name, shell=True)
			while cmd_register.poll() is None:
				pass
				
			train_now = QMessageBox.question(self, 'CDS-1', 'Do you want to retrain our model?', QMessageBox.Yes, QMessageBox.No)
			
			if train_now==QMessageBox.Yes:
				# Retrain
				print("Training our model...")
				cmd_train = subprocess.Popen("python train.py")
				while cmd_train.poll() is None:
					pass
			else:
				# Delete
				print("Reverting registration files.")
				shutil.rmtree("datasets\\"+dataset_name)
		except Exception as error:
			print("Error: "+str(error))
			
		print("Done.")
		exit(1)
	
	def stopRegistration(self):
		pass
		
	def updateDirectoryTextBox(self):
		fname = self.form_registerProperties.txt_fname.text().lower()
		lname = self.form_registerProperties.txt_lname.text().lower()
		directory = "datasets\\"+fname+"-"+lname
		self.form_registerProperties.txt_directory.setText(directory)

				
	def notify(self, id_window):
		self.lock.acquire()
		try:
			self.gsm.sendMessage("09556342339", "ALTECHNOLOGIES\n"+"CAMERA NO: " + str(id_window+1) + "\nUNKNOWN INTRUDER!")
		except Exception as error:
			pass
			
		try:
			self.indicator.write(b"CAM"+str(id_window).encode()+b"\n")
		except Exception as error:
			pass
			
		self.lock.release()
		
	def uploadManager(self):
		while not self.stop_upload:
			try:
				if len(self.upload_processes)==0:
					time.sleep(1)
					continue
				target_id = "1KOPaoV8aSoEqrru__1afLjJY7cYFdJ8H" # Default unknown
				for process in self.upload_processes:
					path = process.get_path()
					filename = path.split("/")[1]
					identity = filename.split("-")[0]
					if identity=="UNKNOWN":
						target_id = "1KOPaoV8aSoEqrru__1afLjJY7cYFdJ8H"
					else:
						target_id = "1zUBeAaaooz-HdbYXeS5qi4H8v34ItzDD"
						
					try:
						file_metadata = {'name': filename, 'parents':[target_id]}
						media = MediaFileUpload(path, mimetype='image/jpeg')
						file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
						print(filename + " uploaded.")
					except Exception as error:
						print(">>>Error occured: "+str(error))
					self.upload_processes.remove(process)
			except Exception as error:
				print(error)
			
	@pyqtSlot(object, object, int)
	def onNewResult(self, frame, identity, index):
		dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S') # PST
		filename = identity.upper()+"-"+dt+"-IMG.jpg"
		if identity=="unknown":
			path = "UNAUTHORIZED/"+filename
		else:
			path = "AUTHORIZED/"+filename
			
		cv.imwrite(path, frame)
		element = UploadElement(path)
		self.upload_processes.append(element)
		del element
		if identity=="unknown":
			Thread(target=self.notify, args=[index]).start()
			
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