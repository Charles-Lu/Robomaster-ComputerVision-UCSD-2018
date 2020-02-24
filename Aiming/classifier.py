import glob
import os.path as path
import pickle
import time

import cv2
import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, directory="data"):
        self.red_file = glob.glob(path.join(directory, "red", "*.jpg"))
        # blue
        self.negative_file = glob.glob(path.join(directory, "negative", "*.jpg"))

    def load(self):
        x = []
        y = []
        for file in self.red_file:
            img = cv2.imread(file)
            x.append(img)
            y.append(1)
        for file in self.negative_file:
            img = cv2.imread(file)
            x.append(img)
            y.append(0)
        x = np.array(x)
        y = np.array(y)
        return x, y


class CNN:
    def __init__(self, network="MobileNet"):
        self.network = network
        x, y = Data("../aiming/data").load()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=0,
                                                                                test_size=0.15, shuffle=True)
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)

        if self.network == "MobileNet":
            self.model = MobileNetV2(input_shape=(96, 96, 3), weights=None, classes=3, alpha=0.5)
        elif self.network == "DenseNet":
            self.model = DenseNet121(input_shape=(96, 96, 3), weights=None, classes=3)
        sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train(self, mode="load"):
        if mode == "train":
            tb_cb = TensorBoard(log_dir='logs', histogram_freq=0)
            change_lr = ReduceLROnPlateau(factor=0.667, patience=5)
            ckpt = ModelCheckpoint('mobile_0.5_ckpt{epoch}.h5', save_best_only=False, mode='auto', period=1)
            cbks = [change_lr, tb_cb, ckpt]
            datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05,
                                         fill_mode='nearest', shear_range=0.05)
            datagen.fit(self.x_train)
            # self.model.load_weights("mobile_ckpt47.h5")
            self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=64),
                                     steps_per_epoch=self.x_train.shape[0] // 64,
                                     epochs=100,
                                     callbacks=cbks,
                                     validation_data=(self.x_test, self.y_test),
                                     shuffle=True,
                                     verbose=1,
                                     initial_epoch=0)
        elif mode == "load":
            if self.network == "MobileNet":
                self.model.load_weights("../aiming/mobile_0.5_ckpt43.h5")
            elif self.network == "DenseNet":
                self.model.load_weights("ckpt14.h5")

    def predict(self, images):
        img_arr = np.array(images).reshape((len(images), 96, 96, 3))
        # pre_time = time.time()
        prob = self.model.predict(img_arr, batch_size=len(images), verbose=0)
        # print("inference time:", time.time()-pre_time)
        prob = np.array(prob)
        return prob.argmax(1), prob


if __name__ == '__main__':
    clf = CNN()
    clf.train("train")
    # clf.train()
    # image = cv2.imread("data/red/1564129702.0663838.jpg")
    # t0 = time.time()
    # print(clf.predict(image))
    # print(time.time() - t0)
