# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import math
import imutils
import time
import cv2
import os.path

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

## Get Proto Models
faceProto="dist/models/opencv_face_detector.pbtxt"
faceModel="dist/models/opencv_face_detector_uint8.pb"
ageProto="dist/models/age_deploy.prototxt"
ageModel="dist/models/age_net.caffemodel"
genderProto="dist/models/gender_deploy.prototxt"
genderModel="dist/models/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

padding=20

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("dist/models/deploy.prototxt.txt", "dist/models/res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		text = ""
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		if confidence >= 0.95:
				# cv2.imwrite("tmp.png", frame)
				resultImg,faceBoxes=highlightFace(faceNet,frame)
				if not faceBoxes:
					print("No face detected")

				for faceBox in faceBoxes:
					face=frame[max(0,faceBox[1]-padding):
							min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
							:min(faceBox[2]+padding, frame.shape[1]-1)]

					blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
					genderNet.setInput(blob)
					genderPreds=genderNet.forward()
					gender=genderList[genderPreds[0].argmax()]

					ageNet.setInput(blob)
					agePreds=ageNet.forward()
					age=ageList[agePreds[0].argmax()]

					# draw the bounding box of the face along with the associated
					# probability
					text = f'{gender}' + ' ' + f'{age[1:-1]}' + ' {:.2f} %'.format(confidence * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						(0, 0, 255), 2)
					cv2.putText(frame, text, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
				saved_conf = confidence
		else:
			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()