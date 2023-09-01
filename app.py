from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


class Face_detection_engine:
    def __init__(self, cascadepath):
        self.face_cascade = cv2.CascadeClassifier(cascadepath)
    
    def face_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )
        #print("The number of faces found = ", len(faces))
        return faces

    def face_capture(self, file_path):
        image =   cv2.imread(file_path)
        return self.face_detect(image), image
    
    def webcam_capture(self):
        cam = cv2.VideoCapture(0)
    
        while True:
            suscess, frame = cam.read()
            if not suscess:
                break
            
            faces = self.face_detect(frame)
            #numfaces = len(faces)
            #print("The number of faces found = ", len(faces))
            #create box
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        cam.release()
        
        
        


   
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/webcam_stream')
def webcam_stream():
    return Response(face_detector.webcam_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload_image', methods = ['GET','POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file path"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Not Select file "}), 400 
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        #face detection with upload file
        face_detector = Face_detection_engine('haarcascade_frontalface_default.xml')
        faces, image = face_detector.face_capture(file_path)

        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', image)
            image_bytes = jpeg.tobytes()
        
        num_faces = len(faces)
        print("The number of faces found = ", len(faces))
        
        return Response(image_bytes, mimetype='image/jpeg')



if __name__ =="__main__":
    
    face_detector = Face_detection_engine('haarcascade_frontalface_default.xml')
    
    app.run(debug=True)
    
    