import face_recognition
from face_recognition.api import face_locations
from sklearn import svm
import os
from joblib import dump, load
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

classifier = svm.SVC(gamma="scale")
classifier.fit(encodings,names)

dump(classifier, "classifier.joblib")

"""
test_image = face_recognition.load_image_file("modi.jpg")

face_locations = face_recognition.face_locations(test_image)
n = len(face_locations)
print(f"Number of faces detected : {n}")

for i in range(n):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = classifier.predict([test_image_enc])
    print(*name)
"""