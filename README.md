# CodeClauseInternship_facedetection
!pip install opencv-python


import cv2
print(cv2.__version__)

face_cascade = cv2.CascadeClassifier('haarcascade.xml')


cap = cv2.VideoCapture(0)  # 0 refers to the default camera


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






