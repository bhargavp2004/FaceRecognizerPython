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

def saveEncodings(request):
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

def findFace(request, image) :
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    face_landmarks_list = face_recognition.face_landmarks(image)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                cv2.circle(image, point, 2, (0, 255, 0), 2) 
    
    cv2.namedWindow("Faces", cv2.WINDOW_NORMAL)
    cv2.imshow("Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    face_encodings = face_recognition.face_encodings(image)[0]


    file = open(r'D:\B.TECH\6TH SEMESTER\Project\FaceRecognizerPython\Encodings.p', 'rb')
    encodeListWithStudentIds = pickle.load(file)
    for encoding, student_id in encodeListWithStudentIds:
        studentsIds.append(student_id)
        encodeList.append(encoding)

    match = face_recognition.compare_faces(encodeList, face_encodings)
    print(match)
    if match.count(True) :
        idx = match.index(True)
        return idx
    return -1

def main(request) :
    image = cv2.imread(r'D:\B.TECH\6TH SEMESTER\Project\FaceRecognizerPython\ImagesToCheck\ak.jpg')
    idx = findFace(request, image)
    if str(idx) == str(-1) :
        return HttpResponse("No matching faces")
    studentId = studentsIds[idx]
    return HttpResponse(studentId)

