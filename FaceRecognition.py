import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
akbar_image = face_recognition.load_image_file("ImageBasic/Akbar_Zyarif_Teguh/Akbar_Zyarif_Teguh_01.jpg")
akbar_face_encoding = face_recognition.face_encodings(akbar_image)[0]

# Load a second sample picture and learn how to recognize it.
arif_image = face_recognition.load_image_file("ImageBasic/Arif_Budi_Almawan/Arif_Budi_Almawan_01.jpg")
arif_face_encoding = face_recognition.face_encodings(arif_image)[0]

# Load a second sample picture and learn how to recognize it.
iyasa_image = face_recognition.load_image_file("ImageBasic/Iyasa_Irfadana/Iyasa_Irfadana_01.jpg")
iyasa_face_encoding = face_recognition.face_encodings(iyasa_image)[0]

# Load a second sample picture and learn how to recognize it.
izza_image = face_recognition.load_image_file("ImageBasic/Izza_Latifatul_Muna/Izza_Latifatul_Muna_01.jpg")
izza_face_encoding = face_recognition.face_encodings(izza_image)[0]

# Load a second sample picture and learn how to recognize it.
karin_image = face_recognition.load_image_file("ImageBasic/Karin_Mayludya_Handi/Karin_Mayludya_Handi_01.jpg")
karin_face_encoding = face_recognition.face_encodings(karin_image)[0]

# Load a second sample picture and learn how to recognize it.
novri_image = face_recognition.load_image_file("ImageBasic/Novri_Lukman_Zyarif/Novri_Lukman_Zyarif_01.jpg")
novri_face_encoding = face_recognition.face_encodings(novri_image)[0]

# Load a second sample picture and learn how to recognize it.
santi_image = face_recognition.load_image_file("ImageBasic/Santi_Nanda_Putri/Santi_Nanda_Putri_01.jpg")
santi_face_encoding = face_recognition.face_encodings(santi_image)[0]

# Load a second sample picture and learn how to recognize it.
solichah_image = face_recognition.load_image_file("ImageBasic/Solichah_Alma_Kurniawati/Solichah_Alma_Kurniawati_01.jpg")
solichah_face_encoding = face_recognition.face_encodings(solichah_image)[0]

# Load a second sample picture and learn how to recognize it.
tica_image = face_recognition.load_image_file("ImageBasic/Tica_Laudita_Nabilah/Tica_Laudita_Nabilah_01.jpg")
tica_face_encoding = face_recognition.face_encodings(tica_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    akbar_face_encoding,
    arif_face_encoding,
    iyasa_face_encoding,
    izza_face_encoding,
    karin_face_encoding,
    novri_face_encoding,
    santi_face_encoding,
    solichah_face_encoding,
    tica_face_encoding
]
known_face_names = [
    "Akbar Zyarif Teguh",
    "Arif Budi Almawan",
    "Iyasa Irfadana",
    "Izza Latifatul Muna",
    "Karin Mayludya Handi",
    "Novri Lukman Zyarif",
    "Santi Nanda Putri",
    "Solichah Alma Kurniawati",
    "Tica Laudita Nabilah"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()