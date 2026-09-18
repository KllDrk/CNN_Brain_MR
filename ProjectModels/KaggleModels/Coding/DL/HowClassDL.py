from __future__ import print_function
import os
from pathlib import Path

import keras
from keras.layers import Dense
import tensorflow as tf
from keras import layers, models
from keras.layers import *
from keras.layers.preprocessing.image_preprocessing import Rescaling
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 40
NUM_CLASSES = 2
CHANNELS = 3


def get_dataset_partitions_tf(ds, train_split=0.7, val_split=0.2, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    print(len(ds))
    train_size = int(train_split * ds_size)
    print(train_size)
    val_size = int(val_split * ds_size)
    int(test_split * ds_size)

    train_ds = ds.take(train_size)

    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def plot_history(hists, attribute='val_loss', axis=(-1, 21, 0.40, 0.80), loc='lower right'):
    plt.figure(figsize=(12, 8))
    plt.axis(axis)
    plt.plot(hists.history[attribute])
    plt.title(attribute)
    plt.ylabel(attribute[-3:])
    plt.xlabel('epoch')
    plt.legend(['ReLU'], loc=loc)
    plt.show()


def ClassTypeDLCreate():
    Url = str(Path.cwd().parent.parent)
    print(Url)
    modelsUrl = f"{Url}/Models/"
    datasetUrl = f"{Url}/Datasets/"
    DLClassTypeUrl = "processedDataset/"

    os.chdir(f"{datasetUrl}{DLClassTypeUrl}")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "ClassType",
        shuffle=True,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    print(class_names)
    print(len(dataset))
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    print(len(train_ds))
    print(len(test_ds))
    print(len(val_ds))

    resize_and_rescale = keras.models.Sequential([
        keras.layers.preprocessing.image_preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        keras.layers.preprocessing.image_preprocessing.Rescaling(1.0 / 255)
    ])

    input_shape = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS)

    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.20),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.20),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.20),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.20),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        Dropout(0.20),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax'),

    ])

    model.build(input_shape=input_shape)
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="Adam", metrics=["accuracy"])

    hist_Relu = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    scores = model.evaluate(test_ds)

    print(scores)
    print(hist_Relu)
    print(hist_Relu.params)
    print(hist_Relu.history.keys())
    print(len(hist_Relu.history['accuracy']))

    acc_Relu = hist_Relu.history['accuracy']
    val_acc_Relu = hist_Relu.history['val_accuracy']
    test_acc_Relu = scores[1]

    loss_Relu = hist_Relu.history['loss']
    val_loss_Relu = hist_Relu.history['val_loss']
    test_loss_Relu = scores[0]

    plot_history(hist_Relu, attribute='accuracy', axis=(0, EPOCHS, 0, 1), loc='lower right')
    plot_history(hist_Relu, attribute='loss', axis=(0, EPOCHS, 0, 1.0), loc='lower right')
    plot_history(hist_Relu, attribute='val_accuracy', axis=(0, EPOCHS, 0, 1), loc='lower right')
    plot_history(hist_Relu, attribute='val_loss', axis=(0, EPOCHS, 0, 1), loc='lower right')

    print(os.listdir(f"{modelsUrl}ModelDL/ClassTypeModel"))

    model_version = max([int(i) for i in os.listdir(f"{modelsUrl}ModelDL/ClassTypeModel") + [0]]) + 1
    model.save(f"{modelsUrl}ModelDL/ClassTypeModel/ClassTypeModel{model_version}")
    pass


def HowClassDLStarter():
    ClassTypeDLCreate()
    pass
