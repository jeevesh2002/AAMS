import face_recognition
from face_recognition.api import face_locations
from sklearn import svm
import os

encodings = []
names = []
# Training Directory
train_dir = os.listdir("assets")

for student in train_dir:
    pix = os.listdir("assets/" + student)

    for student_image in pix:
        face = face_recognition.load_image_file("assets/" + student + "/" + student_image)
        face_bounding_boxes = face_recognition.face_locations(face)

        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]

            encodings.append(face_enc)
            names.append(student)
        else:
            print(student+"/"+student_image+"was skipped")

classifier = svm.SVC(gamma="scale", decision_function_shape="ovr", break_ties=True)
classifier.fit(encodings,names)


test_image = face_recognition.load_image_file("bidentest.jpg")

face_locations = face_recognition.face_locations(test_image)
n = len(face_locations)
print(f"Number of faces detected : {n}")
students_list = os.listdir("assets/")
for i in range(n):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = classifier.predict([test_image_enc])
    print(*name)
    print(test_image_enc)
    #print(classifier.decision_function(test_image_enc),[students_list.index(f"{name[0]}")])
    #y = classifier.decision_function([test_image_enc])
    #print(y)
