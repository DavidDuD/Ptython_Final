from flask import Flask, render_template, redirect, request
import base64, urllib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import pandas as pd

model = load_model('emotion_detection_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
df = pd.read_excel("music.xlsx")

def get_music(t):
    d = df[df["type"]==t]
    d1 = d.values[:, 2:].tolist()
    col = d.columns.tolist()[2:]
    res = []
    
    for i in d1:
        res.append({i:j for i, j in zip(col, i)})
    
    return res

def get_face(img):
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
    
    x, y, w, h = tuple(faces[0])
    
    img = img[y:y+h, x:x+w]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite("r.png", img)
    
    return img

def predict(img):
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    im = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    #im = cv2.imread("test.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    im1 = get_face(img)
    im1 = np.array([cv2.resize(im1,(48, 48),interpolation=cv2.INTER_CUBIC)])
    
    im = np.array([cv2.resize(im,(48, 48),interpolation=cv2.INTER_CUBIC)])
    
    res = model.predict(im1)[0].tolist()
    result = labels[res.index(max(res))]
    
    return result

def base64_to_image(base64_string, file_name):
    try:
        with open(file_name, "wb") as file:
            #print(urllib.parse.unquote(base64_string).encode())
            img = base64.urlsafe_b64decode(urllib.parse.unquote(base64_string).split(",")[1])
            file.write(img)
        print(f"Change base64 to image: {file_name}")
        return img
    except Exception as e:
        print(f"Change base64 to image faild:{e}")
        return None

# Start flask app
# --------------------------------------------------------------------------------------------------

app = Flask(__name__, static_folder='dist', template_folder="template")

@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/upload", methods=["POST"])
def pre():
    if request.method == "POST":
        img = request.values["img"]
        img = img.replace(" ", "+")
        missing_padding = 4 - len(img) % 4
        if missing_padding:
            img += '=' * missing_padding
        img = base64_to_image(img, "file.png")
        if img!=None:
            try:
                result = predict(img)
            except:
                return {"result":"No face detected, prediction failed!", "music":"None"}
            music = get_music(result)
            return {"result":f"Current detected emotion is {result}.", "music":music}
        else:
            return {"result":"Current no detected emotion!", "music":[]}
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)