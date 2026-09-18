import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import pandas as pd
import cv2 as cv2
import numpy as np
from tqdm import tqdm
import subprocess

subprocess.run(["python", "UrlFind.py"])

datasetUrl = "C:/Users/ahmet/Desktop/FinalProjesi/ProjectModels/KaggleModels/Datasets/"
rawTumorDataset = "RawDataset/DatasetTumor/kaggle_3m/"
processedTumorDatasetUrl = "processedDataset/DatasetTumor/"

# print(os.listdir(
#    f"C:/Users/ahmet/Desktop/FinalProjesi/ProjectModels/KaggleModels/Datasets/RawDataset/DatasetAlzheimer/Dataset"
#    f"/Mild_Demented"))

data_path = []
for sub_dir_path in tqdm(glob.glob(f"{datasetUrl}{rawTumorDataset}" + "*")):
    try:
        dir_name = sub_dir_path.split('/')[-1]
        for filename in os.listdir(sub_dir_path):
            mask_path = sub_dir_path + '/' + filename
            data_path.extend([dir_name, mask_path])
    except Exception as e:
        print(e)

filenames = data_path[::2]
masks = data_path[1::2]

df = pd.DataFrame(data={"patient_id": filenames, "img_path": masks})

original_img = df[~df['img_path'].str.contains("mask")]
mask_img = df[df['img_path'].str.contains("mask")]

os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}NormalImg/img/")
for i in tqdm(original_img.img_path):
    img = cv2.imread(i)
    fileName = i.split('/')[-1]
    cv2.imwrite(fileName, img)

os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}NormalImg/mask/")
for i in tqdm(mask_img.img_path):
    img = cv2.imread(i)
    fileName = i.split('/')[-1]
    cv2.imwrite(fileName, img)

data_path_img = []
for n in tqdm(glob.glob(f"{datasetUrl}{processedTumorDatasetUrl}NormalImg/img/" + "*")):
    name_file = (n.split('/')[-1]).split('\\')
    name_file2 = (n.split('/')[0:-1])
    nameX = ""
    for name in name_file2:
        nameX += name + '/'
    nameX += name_file[0] + '/' + name_file[1]
    name_file_url = name_file[1].split('.')[0]
    data_path_img.append(f"{datasetUrl}{processedTumorDatasetUrl}ConvImg/img/" + name_file_url + ".png")
    os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}ConvImg/img/")
    img = cv2.imread(nameX)
    cv2.imwrite(name_file_url + '.png', img)

data_path_mask = []
for n in tqdm(glob.glob(f"{datasetUrl}{processedTumorDatasetUrl}NormalImg/mask/" + "*")):
    name_file = (n.split('/')[-1]).split('\\')
    name_file2 = (n.split('/')[0:-1])
    nameX = ""
    for name in name_file2:
        nameX += name + '/'
    nameX += name_file[0] + '/' + name_file[1]
    name_file_url = name_file[1].split('.')[0]
    data_path_mask.append(f"{datasetUrl}{processedTumorDatasetUrl}ConvImg/mask/" + name_file_url + ".png")
    os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}ConvImg/mask/")
    img = cv2.imread(nameX)
    cv2.imwrite(name_file_url + '.png', img)

i = 0
folderStr = f"{datasetUrl}{processedTumorDatasetUrl}"
for mask_path in tqdm(data_path_mask):
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
