import tensorflow as tf
from engine import facenet
from mtcnn.mtcnn import MTCNN
import cv2 as cv
import numpy as np
import os
import vlc
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

font = cv.FONT_HERSHEY_SIMPLEX
facenet.load_model("models/20180402-114759/20180402-114759.pb")
directory = "datasets"
try:
    name = sys.argv[1]
except Exception as error:
    print("Please provide name argument.")
    exit()
if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(directory+"/"+name):
    os.mkdir(directory+"/"+str(name))

ds_path = directory+"/"+name
print("Generated path: "+ds_path)
detector = MTCNN()

with tf.Session().as_default() as sess:
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    i = 1
    max = 200
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        try:
            faces = detector.detect_faces(frame)
            for face in faces:
                bounding_box = face['box']
                keypoints = face['keypoints']
                face_patch = frame[bounding_box[1]:bounding_box[1]+bounding_box[3],
                                   bounding_box[0]:bounding_box[0]+bounding_box[2], :]
                face_patch = cv.resize(face_patch, (150, 150))
                face_patch = np.stack(face_patch)
                cv.rectangle(frame, (bounding_box[0], bounding_box[1]), (
                    bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 155, 255), 2)
                feed_dict = {images_placeholder: [
                    face_patch], phase_train_placeholder: False}
                init = embeddings.global_variables_initializer()
                embs = sess.run(init, feed_dict=feed_dict)
                if len(embs[0]) == 512:
                    if i == (max+1):
                        print("Done.")
                        exit(0)
                    cv.imwrite(ds_path+"/img-"+name+"-" +
                               str(i)+".jpg", face_patch)
                    print("Gathered sample image: "+str(i)+"/"+str(max))
                    i += 1

        except Exception as error:
            print(error)
        frame = cv.resize(frame, (780, 480))
        cv.imshow("Capturing Face Signatures", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
