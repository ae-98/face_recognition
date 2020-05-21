
import face_recognition
import cv2


# utilisation camera 1
video_capture = cv2.VideoCapture(0)

# Load une image et encoder
#obama_image = face_recognition.load_image_file("obama.jpg")
#obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load une autre image et encoder
#biden_image = face_recognition.load_image_file("biden.jpg")
#biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load une image et encoder.
errkhis_image = face_recognition.load_image_file("ma_photo.jpg")
errkhis_face_encoding = face_recognition.face_encodings(errkhis_image)[0]

# Creation des arrays des visages encoder et un autre array pour leurs noms
known_face_encodings = [
#    obama_face_encoding,
 #   biden_face_encoding,
    errkhis_face_encoding
]
known_face_names = [
#    "Barack Obama",
#    "Joe Biden",
    "Ayoub Errkhis"
]

# Initialisation des variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # lire un frame depuis la camera
    ret, frame = video_capture.read()

    # Reduire la taille des frame pour avoir une lecture rapide depuis la camera
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # convertion de BGR ( OpenCV ) a RGB  ( face_recognition )
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # trouver le visage et l encoder depuis le frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # comparer entre les visage s ils ne sont pas identique
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # comparer s ils sont identiques
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # afficher les resultats
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # retailler vers la taille originale
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # dessiner un rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # dessiner le nom de la personne
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # afficher les resultats ss forme video
    cv2.imshow('Video', frame)

    # cliquer sur q pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release les sources
video_capture.release()
cv2.destroyAllWindows()
