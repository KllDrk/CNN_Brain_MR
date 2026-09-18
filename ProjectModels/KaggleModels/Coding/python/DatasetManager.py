import os
import glob
import pandas as pd
import cv2 as cv2
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image

from ProjectModels.KaggleModels.Coding.python.UrlFind import urlFind

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mainUrlFolder = urlFind()
datasetUrl = f"{mainUrlFolder}ProjectModels\\KaggleModels\\Datasets\\"
rawTumorDataset = "RawDataset\\DatasetTumor\\kaggle_3m\\"
rawAlzheimerDataset = "RawDataset\\DatasetAlzheimer\\Dataset\\"
processedTumorDatasetUrl = "processedDataset\\DatasetTumor\\"
processedAlzheimerDatasetUrl = "processedDataset\\DatasetAlzheimer\\Dataset\\"
processedClassTypeDatasetUrl = "processedDataset\\ClassType\\"


def ProcessingTheRawTumorDataset():
    data_path = []
    for sub_dir_path in tqdm(glob.glob(f"{datasetUrl}{rawTumorDataset}" + "*"), desc="ProcessingTheRawTumorDataset:"):
        try:
            dir_name = sub_dir_path.split('/')[-1]
            for filename in os.listdir(sub_dir_path):
                mask_path = sub_dir_path + '/' + filename
                data_path.extend([dir_name, mask_path])
        except Exception as e:
            print(e)
    return data_path


def SplittingTheDatasetIntoOriginalAndMask(data_path):
    filenames = data_path[::2]
    masks = data_path[1::2]

    df = pd.DataFrame(data={"patient_id": filenames, "img_path": masks})

    original_img = df[~df['img_path'].str.contains("mask")]
    mask_img = df[df['img_path'].str.contains("mask")]
    return original_img, mask_img


def SaveDirectoryNormalImage(img_path, img_class):
    os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}NormalImg/{img_class}/")
    for i in tqdm(img_path.img_path, desc=f"SaveDirectoryNormalImage[{img_class}]:"):
        img = cv2.imread(i)
        fileName = i.split('/')[-1]
        cv2.imwrite(fileName, img)
    pass


def ConvertImageTifToPng(img_class):
    data_path = []
    for n in tqdm(glob.glob(f"{datasetUrl}{processedTumorDatasetUrl}NormalImg/{img_class}/" + "*"),
                  desc=f'ConvertImageTifToPng[{img_class}]:'):
        name_file = (n.split('/')[-1]).split('\\')
        name_file2 = (n.split('/')[0:-1])
        nameX = ""
        for name in name_file2:
            nameX += name + '/'
        nameX += name_file[0] + '/' + name_file[1]
        name_file_url = name_file[1].split('.')[0]
        data_path.append(f"{datasetUrl}{processedTumorDatasetUrl}ConvImg/{img_class}/" + name_file_url + ".png")
        os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}ConvImg/{img_class}/")
        img = cv2.imread(nameX)
        cv2.imwrite(name_file_url + '.png', img)
    return data_path


def ClassTumorDataset(data_path_mask):
    folderStr = f"{datasetUrl}{processedTumorDatasetUrl}"
    for mask_path in tqdm(data_path_mask, desc=f"ClassTumorDataset:"):
        img = cv2.imread(folderStr + "ConvImg/img/" + mask_path.split('/')[-1][0:-9] + ".png")
        mask = cv2.imread(mask_path)
        if np.max(cv2.imread(mask_path)) == 255:
            os.chdir(str(folderStr) + "ClassImg/img/TumorImg/")
            string = mask_path.split('/')[-1][0:-9] + ".png"
            cv2.imwrite(string, img)
            os.chdir(str(folderStr) + "ClassImg/mask/TumorImg/")
            string = mask_path.split('/')[-1]
            cv2.imwrite(string, mask)
        else:
            os.chdir(str(folderStr) + "ClassImg/img/NormalImg/")
            string = mask_path.split('/')[-1][0:-9] + ".png"
            cv2.imwrite(string, img)
            os.chdir(str(folderStr) + "ClassImg/mask/NormalImg/")
            string = mask_path.split('/')[-1]
            cv2.imwrite(string, mask)
    pass


def BrainTumorDatasetClass():
    data_path = ProcessingTheRawTumorDataset()
    original_img, mask_img = SplittingTheDatasetIntoOriginalAndMask(data_path)
    SaveDirectoryNormalImage(original_img, "img")
    SaveDirectoryNormalImage(mask_img, "mask")
    ConvertImageTifToPng("img")
    data_path_mask = ConvertImageTifToPng("mask")
    ClassTumorDataset(data_path_mask)
    pass


def CopyAlzheimerDataset(img_class):
    origin = f"{datasetUrl}{rawAlzheimerDataset}{img_class}"
    target = f"{datasetUrl}{processedAlzheimerDatasetUrl}{img_class}"
    for file_name in tqdm(os.listdir(origin)):
        shutil.copy(origin + file_name, target + file_name)
    pass


def BrainAlzheimerDatasetClass():
    CopyAlzheimerDataset("Mild_Demented/")
    CopyAlzheimerDataset("Very_Mild_Demented/")
    CopyAlzheimerDataset("Moderate_Demented/")
    CopyAlzheimerDataset("Non_Demented/")
    pass


def CopyDatasetAlzheimerToClassType(dicName):
    origin = f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}"
    target = f"{datasetUrl}{processedClassTypeDatasetUrl}Alzheimer/"
    for file_name in tqdm(os.listdir(origin), desc="CopyDatasetAlzheimerToClass:"):
        shutil.copy(origin + file_name, target + file_name)
    pass
    pass


def CopyDatasetTumorToClassType(dicName):
    origin = f"{datasetUrl}{processedTumorDatasetUrl}{dicName}"
    target = f"{datasetUrl}{processedClassTypeDatasetUrl}Tumor/"
    for file_name in tqdm(os.listdir(origin), desc="CopyDatasetTumorToClass:"):
        shutil.copy(origin + file_name, target + file_name)
    pass
    pass


def ResizeDataset(dicName):
    img_path = f"{datasetUrl}{processedClassTypeDatasetUrl}{dicName}"
    files = os.listdir(img_path)
    extensions = ['jpg', 'jpeg', 'png', 'gif']
    for file in files:
        ext = file.split(".")[-1]
        if ext in extensions:
            im = Image.open(img_path + file)
            im_resized = im.resize((256, 256))
            filepath = img_path + f"{file.split('.')[0]}.png"
            im_resized.save(filepath)
            os.remove(img_path + f"{file.split('.')[0]}.jpg")
    pass


def ConvertDataset(dicName):
    img_path = f"{datasetUrl}{processedClassTypeDatasetUrl}{dicName}"
    files = os.listdir(img_path)
    extensions = ['jpg', 'jpeg', 'png', 'gif']
    for file in files:
        ext = file.split(".")[-1]
        if ext in extensions:
            im = Image.open(img_path + file)
            im_resized = im.convert("RGB")
            filepath = img_path + f"{file.split('.')[0]}.png"
            im_resized.save(filepath)
    pass



def BrainDefaultDatasetClass():
    CopyDatasetTumorToClassType("ClassImg/img/NormalImg/")
    CopyDatasetTumorToClassType("ClassImg/img/TumorImg/")

    CopyDatasetAlzheimerToClassType("Mild_Demented/")
    CopyDatasetAlzheimerToClassType("Very_Mild_Demented/")
    CopyDatasetAlzheimerToClassType("Moderate_Demented/")
    CopyDatasetAlzheimerToClassType("Non_Demented/")

    ResizeDataset("Alzheimer\\")
    ResizeDataset("Tumor\\")

    ConvertDataset("Alzheimer\\")
    ConvertDataset("Tumor\\")
    pass


def DatasetManagerStarter():
    BrainTumorDatasetClass()
    BrainAlzheimerDatasetClass()
    BrainDefaultDatasetClass()
    pass

