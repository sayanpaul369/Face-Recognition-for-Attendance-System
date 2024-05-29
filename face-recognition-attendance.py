import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load image faces
sayan_image = face_recognition.load_image_file("faces/sayan.jpeg")
sayan_encoding = face_recognition.face_encodings(sayan_image)[0]
joy_image = face_recognition.load_image_file("faces/joy.jpeg")
joy_encoding = face_recognition.face_encodings(joy_image)[0]
neel_image = face_recognition.load_image_file("faces/neel.jpeg")
neel_encoding = face_recognition.face_encodings(neel_image)[0]

known_face_encodings = [sayan_encoding, joy_encoding, neel_encoding]
known_face_names = ["Sayan", "joy", "Neel"]

# List of expected students
students = known_face_names.copy()
face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        name = None  # Initialize name variable

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Add the text if a person is present
        if name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            blct = (10, 100)
            fontscale = 1.5
            fontcolor = (255, 0, 0)
            thickness = 3
            linetype = 2
            cv2.putText(frame, name + " Present", blct, font, fontscale, fontcolor, thickness, linetype)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
