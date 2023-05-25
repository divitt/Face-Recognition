import cv2
import numpy as np
import face_recognition

imgdivit= face_recognition.load_image_file('images/divit.jpg')
imgdivit= cv2.cvtColor(imgdivit,cv2.COLOR_BGR2RGB)
imgtest= face_recognition.load_image_file('images/divittest.jpg')
imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

imgjay= face_recognition.load_image_file('images/jay.jpg')
imgjay= cv2.cvtColor(imgjay,cv2.COLOR_BGR2RGB)
imgjaytest= face_recognition.load_image_file('images/jaytest.jpg')
imgjaytest= cv2.cvtColor(imgjaytest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgdivit)[0]
encodedivit=face_recognition.face_encodings(imgdivit)[0]
cv2.rectangle(imgdivit,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLoctest=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

faceLocjay=face_recognition.face_locations(imgjay)[0]
encodejay=face_recognition.face_encodings(imgjay)[0]
cv2.rectangle(imgjay,(faceLocjay[3],faceLocjay[0]),(faceLocjay[1],faceLocjay[2]),(255,0,255),2)
faceLocjaytest=face_recognition.face_locations(imgjaytest)[0]
encodejaytest=face_recognition.face_encodings(imgjaytest)[0]
cv2.rectangle(imgjaytest,(faceLocjaytest[3],faceLocjaytest[0]),(faceLocjaytest[1],faceLocjaytest[2]),(255,0,255),2)



results=face_recognition.compare_faces([encodedivit],encodetest)
faceDis=face_recognition.face_distance([encodedivit],encodetest)
print(results,faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('divit',imgdivit)
cv2.imshow('divittest',imgtest)
cv2.imshow('jay',imgjay)
cv2.imshow('jaytest',imgjaytest)
cv2.waitKey(0)