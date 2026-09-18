import os
import tensorflow as tf
import shutil
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
datasetUrl = "C:/Users/ahmet/Desktop/FinalProjesi/ProjectModels/KaggleModels/Datasets/"
processedTumorDatasetUrl = "processedDataset/DatasetTumor/"
processedAlzheimerDatasetUrl = "processedDataset/DatasetAlzheimer/Dataset"
DLAlzheimerDatasetUrl = "DL/Alzheimer/"
DLTumorDatasetUrl = "DL/Tumor/"
DLClassDatasetUrl = "DL/ClassDL/"


# Alzheimer Class
def CopyAlzheimerProcessedDatasetToTempDataset(dicName):
    os.chdir(f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}")
    SplitTypes = []
    for file in os.listdir(os.curdir):
        SplitTypes.append(file)
    trainDS = []
    valDS = []
    testDS = []
    for i in range(len(SplitTypes)):
        if i < len(SplitTypes) * 0.7:
            trainDS.append(SplitTypes[i])
        elif len(SplitTypes) * 0.7 <= i <= len(SplitTypes) * 0.9:
            valDS.append(SplitTypes[i])
        else:
            testDS.append(SplitTypes[i])

    trainDSUrl = "Train"
    trainOrigin = f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}"
    trainTarget = f"{datasetUrl}{DLAlzheimerDatasetUrl}{trainDSUrl}{dicName}"
    for file_name in tqdm(trainDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    valDSUrl = "Val"
    trainOrigin = f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}"
    trainTarget = f"{datasetUrl}{DLAlzheimerDatasetUrl}{valDSUrl}{dicName}"
    for file_name in tqdm(valDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    testDSUrl = "Test"
    trainOrigin = f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}"
    trainTarget = f"{datasetUrl}{DLAlzheimerDatasetUrl}{testDSUrl}{dicName}"
    for file_name in tqdm(testDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)
    pass


def createAlzheimerDataset(IMG_SIZE, directName):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        f"{directName}",
        shuffle=True,
        image_size=(IMG_SIZE, IMG_SIZE)
    )
    return dataset


def cloneAlzheimerDatasetDl():
    CopyAlzheimerProcessedDatasetToTempDataset("/Mild_Demented")
    CopyAlzheimerProcessedDatasetToTempDataset("/Very_Mild_Demented")
    CopyAlzheimerProcessedDatasetToTempDataset("/Non_Demented")
    CopyAlzheimerProcessedDatasetToTempDataset("/Moderate_Demented")
    pass


# TumorImg Class
def CopyTumorProcessedDatasetToTempDataset(dicName):
    img = "ClassImg/img"
    os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}")
    SplitTypes = []
    for file in os.listdir(os.curdir):
        SplitTypes.append(file)
    trainDS = []
    valDS = []
    testDS = []
    for i in range(len(SplitTypes)):
        if i < len(SplitTypes) * 0.7:
            trainDS.append(SplitTypes[i])
        elif len(SplitTypes) * 0.7 <= i <= len(SplitTypes) * 0.9:
            valDS.append(SplitTypes[i])
        else:
            testDS.append(SplitTypes[i])

    imgDSUrl = "/img"
    trainDSUrl = "Train"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}"
    trainTarget = f"{datasetUrl}{DLTumorDatasetUrl}{trainDSUrl}{imgDSUrl}{dicName}"
    for file_name in tqdm(trainDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    valDSUrl = "Val"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}"
    trainTarget = f"{datasetUrl}{DLTumorDatasetUrl}{valDSUrl}{imgDSUrl}{dicName}"
    for file_name in tqdm(valDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    testDSUrl = "Test"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}"
    trainTarget = f"{datasetUrl}{DLTumorDatasetUrl}{testDSUrl}{imgDSUrl}{dicName}"
    for file_name in tqdm(testDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    # Mask
    mask = "ClassImg/mask"
    os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}{mask}{dicName}")
    SplitTypes1 = []
    for file in os.listdir(os.curdir):
        SplitTypes1.append(file)
    trainDS = []
    valDS = []
    testDS = []
    for i in range(len(SplitTypes1)):
        if i < len(SplitTypes1) * 0.7:
            trainDS.append(SplitTypes1[i])
        elif len(SplitTypes1) * 0.7 <= i <= len(SplitTypes1) * 0.9:
            valDS.append(SplitTypes1[i])
        else:
            testDS.append(SplitTypes1[i])

    maskDSUrl = "/mask"
    trainDSUrl = "Train"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{mask}{dicName}"
    trainTarget = f"{datasetUrl}{DLTumorDatasetUrl}{trainDSUrl}{maskDSUrl}{dicName}"
    for file_name in tqdm(trainDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    valDSUrl = "Val"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{mask}{dicName}"
    trainTarget = f"{datasetUrl}{DLTumorDatasetUrl}{valDSUrl}{maskDSUrl}{dicName}"
    for file_name in tqdm(valDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    testDSUrl = "Test"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{mask}{dicName}"
    trainTarget = f"{datasetUrl}{DLTumorDatasetUrl}{testDSUrl}{maskDSUrl}{dicName}"
    for file_name in tqdm(testDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)
    pass


def cloneTumorDatasetDL():
    CopyTumorProcessedDatasetToTempDataset("/NormalImg")
    CopyTumorProcessedDatasetToTempDataset("/TumorImg")
    pass


def CopyClassTypeProcessedDatasetToTempAlzheimerDataset(dicName):
    # Alzheimer
    os.chdir(f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}")
    SplitTypes = []
    for file in os.listdir(os.curdir):
        SplitTypes.append(file)
    trainDS = []
    valDS = []
    testDS = []
    for i in range(len(SplitTypes)):
        if i < len(SplitTypes) * 0.7:
            trainDS.append(SplitTypes[i])
        elif len(SplitTypes) * 0.7 <= i <= len(SplitTypes) * 0.9:
            valDS.append(SplitTypes[i])
        else:
            testDS.append(SplitTypes[i])

    alzheimerDSUrl = "/Alzheimer"
    trainDSUrl = "Train"
    trainOrigin = f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}"
    trainTarget = f"{datasetUrl}{DLClassDatasetUrl}{trainDSUrl}{alzheimerDSUrl}"
    for file_name in tqdm(trainDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    valDSUrl = "Val"
    trainOrigin = f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}"
    trainTarget = f"{datasetUrl}{DLClassDatasetUrl}{valDSUrl}{alzheimerDSUrl}"
    for file_name in tqdm(valDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    testDSUrl = "Test"
    trainOrigin = f"{datasetUrl}{processedAlzheimerDatasetUrl}{dicName}"
    trainTarget = f"{datasetUrl}{DLClassDatasetUrl}{testDSUrl}{alzheimerDSUrl}"
    for file_name in tqdm(testDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)
    pass


def CopyClassTypeProcessedDatasetToTempTumorDataset(dicName):
    img = "ClassImg/img"
    os.chdir(f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}")
    SplitTypes = []
    for file in os.listdir(os.curdir):
        SplitTypes.append(file)
    trainDS = []
    valDS = []
    testDS = []
    for i in range(len(SplitTypes)):
        if i < len(SplitTypes) * 0.7:
            trainDS.append(SplitTypes[i])
        elif len(SplitTypes) * 0.7 <= i <= len(SplitTypes) * 0.9:
            valDS.append(SplitTypes[i])
        else:
            testDS.append(SplitTypes[i])

    tumorDSUrl = "/Tumor"
    trainDSUrl = "Train"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}"
    trainTarget = f"{datasetUrl}{DLClassDatasetUrl}{trainDSUrl}{tumorDSUrl}"
    for file_name in tqdm(trainDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    valDSUrl = "Val"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}"
    trainTarget = f"{datasetUrl}{DLClassDatasetUrl}{valDSUrl}{tumorDSUrl}"
    for file_name in tqdm(valDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    testDSUrl = "Test"
    trainOrigin = f"{datasetUrl}{processedTumorDatasetUrl}{img}{dicName}"
    trainTarget = f"{datasetUrl}{DLClassDatasetUrl}{testDSUrl}{tumorDSUrl}"
    for file_name in tqdm(testDS):
        shutil.copy(trainOrigin + "/" + file_name, trainTarget + "/" + file_name)

    pass


def cloneClassTypeDatasetDL():
    cloneAlzheimerDatasetDl()
    cloneTumorDatasetDL()

    CopyClassTypeProcessedDatasetToTempAlzheimerDataset("/Mild_Demented")
    CopyClassTypeProcessedDatasetToTempAlzheimerDataset("/Very_Mild_Demented")
    CopyClassTypeProcessedDatasetToTempAlzheimerDataset("/Non_Demented")
    CopyClassTypeProcessedDatasetToTempAlzheimerDataset("/Moderate_Demented")

    CopyClassTypeProcessedDatasetToTempTumorDataset("/NormalImg")
    CopyClassTypeProcessedDatasetToTempTumorDataset("/TumorImg")
    pass


cloneClassTypeDatasetDL()
