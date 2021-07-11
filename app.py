import face_recognition
from face_recognition.api import face_locations
from numpy import load
from sklearn import svm
import os
from joblib import dump, load


test_image = face_recognition.load_image_file("biden.jpg")

face_locations = face_recognition.face_locations(test_image)
n = len(face_locations)
print(f"Number of faces detected : {n}")

classifier = load("classifier.joblib")
students_list = os.listdir("assets/")

for i in range(n):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = classifier.predict([test_image_enc])
    print(*name)
    print(students_list.index(f"{name[0]}"))
    print(classifier.decision_function([test_image_enc])[0][students_list.index(f"{name[0]}")])
    y = classifier.decision_function([test_image_enc])
    print(y)
    print(type(y))
    print(y[0][2])
