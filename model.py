import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import datetime
import scipy.fftpack
import pandas as pd
import numpy as np
from tqdm import tqdm

#! defining the basic structure of the model
#* this will take a input of 250 sample 
class eegmodel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(125, activation=tf.nn.relu)
        self.dense2 = keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense3 = keras.layers.Dense(80, activation=tf.nn.relu)
        self.dense4 = keras.layers.Dense(50, activation=tf.nn.relu)
        self.dense5 = keras.layers.Dense(25, activation=tf.nn.relu)
        self.dense6 = keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense7 = keras.layers.Dense(6, activation=tf.nn.relu)
        self.dense8 = keras.layers.Dense(4, activation=tf.nn.relu)
        self.out = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        return self.out(x)
    
dftraintotal = pd.read_csv("./data/subj1_series1_data.csv")

#! this code is taking into consideration of only one channel
y = np.array(dftraintotal["Fp1"])

dflabeltotal = pd.read_csv("./data/subj1_series1_events.csv")

#! this code is taking into consideration of only one acivity
ylabel = np.array(dflabeltotal["HandStart"])


num_train_data = 100000
num_test_data = 19496
numinput = 250
num_train_batch = 0

train_data = y[:num_train_data]
train_labels = ylabel[0:num_train_data]

train_data = np.array(train_data)
train_labels = np.array(train_labels)

model = eegmodel()
optimizer = tf.keras.optimizers.Adam()
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae']) 
model.build(input_shape=(None, 250))

#* taking the window to be o 250 samples
train_data = train_data.reshape(400, 250)

#? making the label out of it 
train_labels = train_labels.reshape(400, 250)
finallabel = []
for i in range(train_labels.shape[0]):
    finallabel.append(max(train_labels[i]))

train_labels = finallabel
train_labels = np.array(train_labels)
train_labels = train_labels.reshape(400, 1)

EPOCHS = int(input("Enter the number of epochs: "))
strt_time = datetime.datetime.now()
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                  validation_split=0.2, verbose=2,
                  callbacks=[])
curr_time = datetime.datetime.now()
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()
print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
plt.show()


test_data = y[num_train_data:-246]
test_labels = ylabel[num_train_data:-246]
test_data = test_data.reshape(int(test_data.shape[0]/250), 250)

test_labels = test_labels.reshape(77, 250)
finallabel = []
for i in range(test_labels.shape[0]):
    finallabel.append(max(test_labels[i]))
test_labels = finallabel
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(77, 1)

dnn_predictions = model.predict(test_data)
pred = []
for it in dnn_predictions:
    if it > 0.5:
        it = 1
        pred.append(1)
    else:
        it = 0
        pred.append(0)

print("the pred is:")
for i in range(len(pred)):
    if pred[i] == 1:
        print(i)

print("the actual is:")
for i in range(len(pred)):
    if test_labels[i] == 1:
        print(i)