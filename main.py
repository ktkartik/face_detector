import cv2  

face_cap = cv2.CascadeClassifier(
    "C:/Users/kt089/AppData/Roaming/Python/Python310/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)
print(face_cap.empty())
eyes_cap =cv2.CascadeClassifier(
  "C:/Users/kt089/AppData/Roaming/Python/Python310/site-packages/cv2/data/haarcascade_eye.xml"
)
print(eyes_cap.empty())
video_capture = cv2.VideoCapture(0)

while True:
    ret, video_data = video_capture.read()
    
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    eyes=eyes_cap.detectMultiScale(
        video_data,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            video_data,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
    for (x, y, w, h) in eyes:
        cv2.rectangle(
            video_data,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("kt52", video_data)

    if cv2.waitKey(10) & 0xFF == ord("a"):
        break

video_capture.release()