from flask import Flask, render_template, url_for
app = Flask(__name__)


import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
haar_cascade_face = cv2.CascadeClassifier(r'path/to/haarcascade_frontalface_default.xml')

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def click_photo():
	flag = 0
	cap = cv2.VideoCapture(0)
	i=0
	while i<15:
	    ret, frame = cap.read()
	    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    faces_rects = haar_cascade_face.detectMultiScale(image, scaleFactor = 1.2, minNeighbors = 5)
	    if len(faces_rects)>=1:
	    	flag=1
	    	break
	    time.sleep(1)
	    i+=1
	cv2.destroyAllWindows()
	cap.release()
	return flag


@app.route("/")
@app.route("/home")
def hello():
	return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    status = click_photo()
    if status == 1:
    	message = "Success"
    else:
    	message = "Failure"
    return render_template("result.html", message=message)

if __name__ == '__main__':
	app.run(debug=True)
