import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  
import warnings
from scipy import stats
import matplotlib.pyplot as plt 

warnings.filterwarnings("ignore")

from tensorflow import keras
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

df_train = pd.read_csv("eeg3.csv", header=None)

idx = df_train.shape[1]-1
Y = np.array(df_train[idx].values).astype(np.int8)
Y = Y - 1
X = np.array(df_train[list(range(idx-1))].values)[..., np.newaxis]
X = X.reshape((X.shape[0], X.shape[1]))

nclass = len(np.unique(Y))

inp = Input(shape=(idx-1, 1))
layers = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
layers = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(layers)
layers = MaxPool1D(pool_size=2)(layers)
layers = Dropout(rate=0.1)(layers)
layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
layers = MaxPool1D(pool_size=2)(layers)
layers = Dropout(rate=0.1)(layers)
layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
layers = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(layers)
layers = MaxPool1D(pool_size=2)(layers)
layers = Dropout(rate=0.1)(layers)
layers = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(layers)
layers = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(layers)
layers = GlobalMaxPool1D()(layers)
layers = Dropout(rate=0.2)(layers)

dense_1 = Dense(64, activation=activations.relu, name="dense_1")(layers)
dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam(0.001)

model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
model.summary()
file_path = "cnn_model.h5"
train = 0

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=25, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=25, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=25, verbose=1, callbacks=callbacks_list, validation_split=0.1)
