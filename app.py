from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import datetime
import pandas as pd
import csv

app = Flask(__name__)


@app.route('/home')
def main() :
    return render_template('index.html')


def gen_frames():  

    engine = textSpeach.init()

    def resize(img, size) :
        width = int(img.shape[1]*size)
        height = int(img.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

    path = 'ImageAttendence'
    studentImg = []
    studentName = []
    myList = os.listdir(path)
    for cl in myList :
        curimg = cv2.imread(f'{path}/{cl}')
        studentImg.append(curimg)
        studentName.append(os.path.splitext(cl)[0])

    def findEncoding(images) :
        imgEncodings = []
        for img in images :
            img = resize(img, 0.50)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodeimg = face_rec.face_encodings(img)[0]
            imgEncodings.append(encodeimg)
        return imgEncodings
    def MarkAttendence(name) :
        with open('attendence.csv', 'r+') as f :
            myDatalist = f.readlines()
            nameList = []
            for line in myDatalist :
                entry = line.split(',')
                nameList.append(entry[0])

            if name not in nameList :
                now = datetime.now()
                date = now.strftime('%d-%m-%Y')
                time = now.strftime('%H:%M')
                f.writelines(f'\n{name}, {date}, {time}')
                statment = str('welcome to class' + name)
                engine.say(statment)
                engine.runAndWait()


    EncodeList = findEncoding(studentImg)

    camera = cv2.VideoCapture(0)


    while True:
        success, frame = camera.read()  # read the camera frame

        Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

        facesInFrame = face_rec.face_locations(Smaller_frames)
        encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

        for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
            matches = face_rec.compare_faces(EncodeList, encodeFace)
            facedis = face_rec.face_distance(EncodeList, encodeFace)
            # print(facedis)
            matchIndex = np.argmin(facedis)

            if matches[matchIndex] :
                name = studentName[matchIndex].upper()
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                MarkAttendence(name)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/facedetection')
def facede() : 
    return render_template('face_detection.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

df = pd.read_csv("attendence.csv")
df.to_csv("attendence.csv", index=None)

@app.route('/report')
def report():
    data = pd.read_csv("attendence.csv")
    return render_template('report.html', tables=[data.to_html()], titles = [''])



df = pd.read_csv("attendence.csv")
df.to_csv("attendence.csv", index=None)

@app.route('/delete')
def delete() :
    lines = list()
    rownumbers_to_remove= [2,3,4,5,6,7,8,9,10,11]

    with open('attendence.csv', 'r') as read_file:
        reader = csv.reader(read_file)
        for row_number, row in enumerate(reader, start=1):
            if(row_number not in rownumbers_to_remove):
                lines.append(row)

    with open('attendence.csv', 'w') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(lines)
        
    data = pd.read_csv("attendence.csv")
    return render_template('report.html', tables=[data.to_html()], titles = [''])

if __name__ == "__main__" :
    app.run()