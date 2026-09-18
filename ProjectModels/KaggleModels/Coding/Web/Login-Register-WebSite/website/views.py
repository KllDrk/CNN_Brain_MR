import os
import cv2
from PIL import Image

import numpy as np
import tensorflow as tf
from flask import Blueprint, render_template, request, flash, jsonify, Flask, url_for, redirect
from flask_login import login_required, current_user
from OrnekWebSiteleri.OrnekProje1.website.models import File, UploadFileForm
from werkzeug.utils import secure_filename
from pathlib import Path

from ProjectModels.KaggleModels.Coding.python.DLController import runningFuncStarter
from ProjectModels.KaggleModels.Coding.python.ZipFileManager import StarterExtract

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

views = Blueprint('views', __name__, template_folder='template', static_folder='static')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DicProjectUrl = str(Path.cwd().parent.parent)
modelsUrl = f"{DicProjectUrl}\\Models\\"
webUrl = f"{DicProjectUrl}\\Coding\\Web\\Login-Register-Website\\website"

ModelTumor = tf.keras.models.load_model("C:\\Users\\ahmet\\Documents\\Desktop\\FinalProjesi\\ProjectModels\\KaggleModels\\Models\\ModelDL\\TumorModel\\TumorModel1")
Class_Names_Tumor = ["MormalImg", "TumorImg"]

ModelAlzheimer = tf.keras.models.load_model("C:\\Users\\ahmet\\Documents\\Desktop\\FinalProjesi\\ProjectModels\\KaggleModels\\Models\\ModelDL\\AlzheimerModel\\AlzheimerModel1")
Class_Names_Alzheimer = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]

ModelClassType = tf.keras.models.load_model("C:\\Users\\ahmet\\Documents\\Desktop\\FinalProjesi\\ProjectModels\\KaggleModels\\Models\\ModelDL\\ClassTypeModel\\ClassTypeModel1")
Class_Names_Class_Type = ["Alzheimer", "Tumor"]

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file_as_image(data) -> np.ndarray:
    image = np.array(data)
    return image


resultsBool = False


@views.route('/', methods=['GET', 'POST'])
@views.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    form = UploadFileForm()
    websiteUrl = os.path.abspath(os.path.dirname(__file__))
    websiteUrl += "\\static\\files"
    if form.validate_on_submit() and request.method == 'POST':
        file = form.file.data
        print(type(file))
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        print("kayıt edildi.")
        StarterExtract(websiteUrl,
                       "Datasets",
                       str(file.filename))
        print("Dosya Çıkartıldı.")
        runningFuncStarter()
    return render_template("home.html", user=current_user, form=form)


@views.route('/home2', methods=['GET', 'POST'])
@login_required
def home2():
    ResultText = ["", "", "", ""]
    global resultsBool
    websiteUrl = str(os.path.abspath(os.path.dirname(__file__)))
    websiteUrl += "\\static\\files\\"
    print(os.path.abspath(os.path.dirname(__file__)))
    print(request.method)
    if request.method == 'POST':
        resultsBool = True
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                   app.config['UPLOAD_FOLDER'],
                                   secure_filename(file.filename)))
            img = Image.open(websiteUrl + str(file.filename))
            image = np.array(img)
            print(image.shape)
            extensions = ['jpg', 'jpeg', 'png', 'gif']
            ext = file.filename.split(".")[-1]
            print("urlImage2" + websiteUrl)
            print("file:" + file.filename)
            Result = ""
            filePath = str(websiteUrl + file.filename)
            if ext in extensions:
                im = Image.open(websiteUrl + file.filename)
                im_resized = im.resize((256, 256))
                filepath = websiteUrl + f"{file.filename.split('.')[0]}.png"
                im_resized.save(filepath)
                im = Image.open(filepath)
                im_resized = im.convert("RGB")
                filepath = websiteUrl + f"{file.filename.split('.')[0]}.png"
                im_resized.save(filepath)
                img = Image.open(filepath)
                image =np.array(img)
                print("last image:" + str(image.shape))
                img_batch = np.expand_dims(image, 0)
                print("last image batch:" + str(img_batch))
                predictions = ModelClassType.predict(img_batch)
                print(predictions)
                index = np.max(predictions[0])
                predicted_class = Class_Names_Class_Type[np.argmax(predictions[0])]
                print(index)
                print(predicted_class)
                Result = predicted_class
                ResultText[0] = str(predictions)
                ResultText[1] = str(predicted_class)
                pass
            if Result == "Alzheimer":
                img = Image.open(filePath)
                image = np.array(img)
                print("last image:" + str(image.shape))
                image = cv2.merge((image, image, image))
                img_batch = np.expand_dims(image, 0)
                print("last image batch:" + str(img_batch))
                predictions = ModelAlzheimer.predict(img_batch)
                print(predictions)
                index = np.max(predictions[0])
                predicted_class = Class_Names_Alzheimer[np.argmax(predictions[0])]
                print(index)
                print(predicted_class)
                ResultText[2] = str(predictions)
                ResultText[3] = str(predicted_class)
                pass
            elif Result == "Tumor":
                img = Image.open(filePath)
                image = np.array(img)
                print("last image:" + str(image.shape))
                img_batch = np.expand_dims(image, 0)
                print("last image batch:" + str(img_batch))
                predictions = ModelTumor.predict(img_batch)
                print(predictions)
                index = np.max(predictions[0])
                predicted_class = Class_Names_Tumor[np.argmax(predictions[0])]
                print(index)
                print(predicted_class)
                ResultText[2] = str(predictions)
                ResultText[3] = str(predicted_class)
                pass
            print("*"*50)
            print(ResultText)
            flash('Image successfully uploaded and displayed below')
    if request.method == 'GET':

        pass
    return render_template('home2.html', user=current_user)


