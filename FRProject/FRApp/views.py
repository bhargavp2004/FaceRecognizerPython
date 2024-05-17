import cv2
from django.http import HttpResponse
import os
import face_recognition
import pickle
import numpy as np

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
    studentsIds = []
    for path in imageList:
        img_path = os.path.join(imageDirectoryPath, path)
        img = cv2.imread(img_path)
        images.append(img)
        studentsIds.append(os.path.splitext(path)[0])
    return images, studentsIds

def findEncodings(request, imagesList):
    eList = []
    for img in imagesList:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        face_landmarks_list = face_recognition.face_landmarks(img)
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                for point in face_landmarks[facial_feature]:
                    cv2.circle(img, point, 2, (0, 255, 0), 2) 
    
        cv2.namedWindow("Faces", cv2.WINDOW_NORMAL)
        cv2.imshow("Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        encode = face_recognition.face_encodings(rgb_img)[0]
        eList.append(encode)

    return eList

def saveEncodings(request):
    imageDirectoryPath = 'D:/B.TECH/6TH SEMESTER/Project/FaceRecognizerPython/Images'
    imagesList, studentsIds = loadImages(request, imageDirectoryPath)
    encodeList = findEncodings(request, imagesList)
    encodeListWithStudentIds = list(zip(encodeList, studentsIds))
    for encoding, student_id in encodeListWithStudentIds:
        print("Student ID:", student_id)
        print("Encoding:", encoding)
    file = open("D:/B.TECH/6TH SEMESTER/Project/FaceRecognizerPython/Encodings.p", 'wb')
    pickle.dump(encodeListWithStudentIds, file)
    file.close()
    return HttpResponse("Encoding Complete")

def loadPickleFileAndReturnEncodeListWithStudentsIds() :
    file = open(r'D:\B.TECH\6TH SEMESTER\Project\FaceRecognizerPython\Encodings.p', 'rb')
    encodeListWithStudentIds = pickle.load(file)
    return encodeListWithStudentIds

def getStudentsIds():
    studentsIds = []
    encodeListWithStudentIds = loadPickleFileAndReturnEncodeListWithStudentsIds()

    for encoding, student_id in encodeListWithStudentIds:
        studentsIds.append(student_id)
    
    return studentsIds

def getEncodings() :
    encodeList = []
    encodeListWithStudentIds = loadPickleFileAndReturnEncodeListWithStudentsIds()

    for encoding, student_id in encodeListWithStudentIds:
        encodeList.append(encoding)
    
    return encodeList

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


    # Code to compare single face found in current image
    # ***************************************************************
    face_encodings = face_recognition.face_encodings(image)[0]
    encodeList = getEncodings()   
    match = face_recognition.compare_faces(encodeList, face_encodings, 0.5)
    print(match)
    if match.count(True) :
        idx = match.index(True)
        return idx
    return -1
    # ***************************************************************



    # Code to compare more than one faces found in current image
    # ***************************************************************
    # indices = []
    # face_encodings = face_recognition.face_encodings(image)
    # encodeList = getEncodings()
    # known_face_encodings = np.array(encodeList)
    # for encoding in face_encodings:
    #     match = (face_recognition.compare_faces(known_face_encodings, np.array(encoding)))
    #     print(match)
    #     if match.count(True) :
    #         idx = match.index(True)
    #         indices.append(idx)

    # return indices
    # ***************************************************************

    

def main(request) :
    image = cv2.imread(r'D:\B.TECH\6TH SEMESTER\Project\FaceRecognizerPython\ImagesToCheck\b3.jpg')
    # Code for more than one faces found in current image
    # ***************************************************************
    # indices = findFace(request, image)
    # studentsFoundWithIds = []
    # if len(indices) :
    #     studentsIds = getStudentsIds()
    #     print("Id of student(s) found in image : " )
    #     for idx in indices : 
    #         studentId = studentsIds[idx]
    #         studentsFoundWithIds.append(studentId)
    #     return HttpResponse(studentsFoundWithIds)
    # else :
    #     return HttpResponse("No Matching Face Found")
    # ***************************************************************
    
    studentsIds = getStudentsIds()
    idx = findFace(request, image)
    if idx == -1 :
        return HttpResponse("No Matching Face Found")
    else :
        id = studentsIds[idx]
        return HttpResponse(id)
    
    

