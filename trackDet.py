import time
import cv2
import dlib
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from imutils.video import VideoStream
from imutils.video import FPS
import imutils


def classify_frame(net, inputQueue, outputQueue):
	# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			frame = inputQueue.get()
			height , width , layers =  frame[1].shape
			newHeight = 300
			newWidth = 300
			frame = cv2.resize(frame[1], (newWidth, newHeight)) 
			blob = cv2.dnn.blobFromImage(frame, 0.007843,(300, 300), 127.5)
 
			# set the blob as input to our deep learning object
			# detector and obtain the detections
			net.setInput(blob)
			detections = net.forward()
 
			# write the detections to the output queue
			outputQueue.put(detections)

def track(img,pt1,pt2,color,val,cam):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	tracker = dlib.correlation_tracker()
	out = []
	#out.append((minx,miny,maxx,maxy))
	out.append((pt1[0],pt1[1],pt2[0],pt2[1]))
	tracker.start_track(img_gray, dlib.rectangle(*out[0]))
	while True:
        # Read frame from device or file
		retval, img = cam.read()
		if(cv2.waitKey(10)==ord('p')):
			break
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#img = cv2.GaussianBlur(img, (15,15), 0)
		# Get the position of the object, draw a 
		# bounding box around it and display it.
		rect = tracker.get_position()
		pt1 = (int(rect.left()), int(rect.top()))
		pt2 = (int(rect.right()), int(rect.bottom()))
		cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
		#cv2.circle(img, pt1, 50, 255)
		#cv2.circle(img, pt2, 50, 255)
		cv2.circle(img, (int(pt1[0] - (pt1[0]-pt2[0])/2), int(pt1[1] - (pt1[1]-pt2[1])/2)), 50, 255)
		cv2.circle(img, (int(650), int(330)), 50, 255)
		print("Target:",int(pt1[0] - (pt1[0]-pt2[0])/2), int(pt1[1] - (pt1[1]-pt2[1])/2))
		print("Target offset:",(650 -int(pt1[0] - (pt1[0]-pt2[0])/2)), (330-int(pt1[1] - (pt1[1]-pt2[1])/2)))
		# Update the tracker  
		tracker.update(img)
		cv2.imshow("Frame", img)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break


def run():
	detFlag = 0
	#Create the VideoCapture object
	cam = cv2.VideoCapture(0)
	start_time = time.time()
	while True:
		#detect if detFlag = 0
		if(detFlag == 0):
			frame = cam.read()
			width = cam.get(3)   # float
			height = cam.get(4) # float	
			#do detection
			print(width,height)
			start = time.time()
			if inputQueue.empty():
				inputQueue.put(frame)
			# if the output queue *is not* empty, grab the detections
			if not outputQueue.empty():
				detections = outputQueue.get()
			else: 
				detections = None

			if detections is not None:
				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated
					# with the prediction
					confidence = detections[0, 0, i, 2]
 
					# filter out weak detections by ensuring the `confidence`
					# is greater than the minimum confidence
					if confidence < 0.99:
						continue

					# otherwise, extract the index of the class label from
					# the `detections`, then compute the (x, y)-coordinates
					# of the bounding box for the object
					idx = int(detections[0, 0, i, 1])
					dims = np.array([width, height, width, height])
					box = detections[0, 0, i, 3:7] * dims
					(startX, startY, endX, endY) = box.astype("int")

					cv2.rectangle(frame[1], (startX, startY), (endX, endY),COLORS[0], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					track(frame[1], (startX, startY), (endX, endY),COLORS[0], 2,cam)
			
			#key = cv2.waitKey(1) & 0xFF
 
			# if the `q` key was pressed, break from the loop
			#if key == ord("q"):
			#	break

		#check confidence
		#check if object is still for 5 seconds
		elapsed_time = time.time() - start_time
		#print(elapsed_time)

if __name__ == "__main__":
	CLASSES = ["person"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
	# initialize the input queue (frames), output queue (detections),
	# and the list of actual detections returned by the child process
	inputQueue = Queue(maxsize=1)
	outputQueue = Queue(maxsize=1)
	detections = None
	p = Process(target=classify_frame, args=(net, inputQueue, outputQueue,))
	p.daemon = True
	p.start()
	run()