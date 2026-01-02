from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

face_cap = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cap = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cap.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            # Face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            # Region of interest (face only)
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            eyes = eye_cap.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    face_color,
                    (ex, ey),
                    (ex+ew, ey+eh),
                    (0, 255, 0),
                    2
                )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(debug=True)
