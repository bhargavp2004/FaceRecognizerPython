import cv2
from django.http import HttpResponse
import os
import face_recognition
import pickle

imageList = []
studentsIds = []
encodeList = []
def OpenWebCam(request) :
    cap = cv2.VideoCapture(0)
    while True :
        success, img = cap.read()
        cv2.imshow("Webcam", img)
        key = cv2.waitKey(1)
        if key == ord('q') :
            break
    cap.release()
    cv2.destroyAllWindows()
    return HttpResponse("We are done.")

def loadImages(request, imageDirectoryPath):
    imageList = os.listdir(imageDirectoryPath)
    images = []
    for path in imageList:
        img_path = os.path.join(imageDirectoryPath, path)
        img = cv2.imread(img_path)
        images.append(img)
        studentsIds.append(os.path.splitext(path)[0])
    return images

def findEncodings(request, imagesList):
    eList = []
    for img in imagesList:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        encode = face_recognition.face_encodings(rgb_img)[0]
        eList.append(encode)
    return eList

def main(request):
    imageDirectoryPath = 'D:/B.TECH/6TH SEMESTER/Project/FaceRecognizerPython/Images'
    imagesList = loadImages(request, imageDirectoryPath)
    encodeList = findEncodings(request, imagesList)
    encodeListWithStudentIds = list(zip(encodeList, studentsIds))
    for encoding, student_id in encodeListWithStudentIds:
        print("Student ID:", student_id)
        print("Encoding:", encoding)
    file = open("D:/B.TECH/6TH SEMESTER/Project/FaceRecognizerPython/Encodings.p", 'wb')
    pickle.dump(encodeListWithStudentIds, file)
    file.close()
    return HttpResponse("Encoding Complete")
