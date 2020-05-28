import tensorflow as tf

import numpy as np
import glob
import io
import os
import math
"""
ftest = open("dataSet/fist.txt","r")
myList = list()
np_array_reshaped = None
#np_examples = numpy.zeros(f)
for line in ftest:
   #myList = line[1:]
  #line = ftest.readline()
  line = line[1:len(line)-2]
  line = line.replace(',','')
  line = line.split(" ")
  numLine = list(map(int,line))
  myList.append(numLine)
  #print(len(numLine))
  #np_array = np.asarray(numLine)
  #np_array_reshaped = np_array.reshape(int (len(numLine)/8),8)
  #np_examples.append(np_array_reshaped)
  #print(np_array_reshaped)
totalSamples = int((len(myList[0])/8))
np_examples = np.empty((len(myList),totalSamples,8),int)
for i in range(len(np_examples)):
  np_array = np.asarray(myList[i])
  np_array_reshaped = np_array.reshape(totalSamples,8)
  np_examples[i] = np_array_reshaped
# for samples in myList:
#   np_array = np.asarray(samples)
#   np_array_reshaped = np_array.reshape(int (len(samples)/8),8)
#   print(np_array_reshaped.shape)
#   np_examples = np.append(np_examples,np_array_reshaped)
# print(len(np_array_reshaped))
print(np_examples.shape)
print(np_examples[9][1][:])
# print(np_array_reshaped[149])
# print(np_array_reshaped.shape)
ftest.close()"""
if __name__ == "__main__":
  working_dir = "dataSet"
  firstSample = True
  data_x = None
  data_x_train = None
  data_x_test = None
  ratio = 0.8
  data_y = None
  data_y_train = None
  data_y_test = None
  gestureOrder = list()
  gestureNum = 0
  myList = list()
  #my_gesture =
  for path in glob.glob(os.path.join(working_dir,"*.txt")):
    with io.open(path, mode="r", encoding="utf-8") as fd:
      np_array_reshaped = None
      for line in fd:
        line = line[1:len(line)-2]
        line = line.replace(',','')
        line = line.split(" ")
        numLine = list(map(int,line))
        myList.append(numLine)
      if(firstSample):
        totalSamples = int((len(myList[0])/8))
        np_examples = np.empty((len(myList),totalSamples,8),int)
      for i in range(len(np_examples)):
        np_array = np.asarray(myList[i])
        np_array_reshaped = np_array.reshape(totalSamples,8)
        np_examples[i] = np_array_reshaped
      np_examples = tf.keras.utils.normalize(np_examples)
      y_label_Class = os.path.basename(fd.name)
      if(firstSample):
        data_x = np.copy(np_examples)
        data_x_train = np.copy(np_examples[0:int(len(np_examples)*ratio)])
        data_x_test = np.copy(np_examples[int(len(np_examples)*ratio):len(np_examples)])
        firstSample = False
      else:
        data_x = np.concatenate((data_x,np_examples),axis=0)
        data_x_train = np.concatenate((data_x_train,np_examples[0:int(len(np_examples)*ratio)]),axis=0)
        data_x_test = np.concatenate((data_x_test,np_examples[int(len(np_examples)*ratio):len(np_examples)]),axis=0)
      gestureOrder.append(y_label_Class)
      #value_label = [len(gestureOrder)] * 10
      #data_y.append(value_label)
      myList.clear()
      print(data_x.shape)
      print(data_x[9][1][:])
  #data_x = tf.keras.utils.normalize(data_x)
  print(data_x[9][1][:])
  print(data_x[19][1][:])
  # print(data_x[29][1][:])
  # print(data_x[39][1][:])
  samplesPerGesture = int(len(data_x)/len(gestureOrder))
  data_y = list()
  data_y_train = list()
  data_y_test = list()
  #data_x_train = np.empty((5,len(data_x[0]),8))
  #data_x_train = np.empty((0,)*3)
  #print(data_x_train)
  #print(data_x_train.shape)
  for label in range(len(gestureOrder)):
    data_y = data_y + ([label] * samplesPerGesture)
    data_y_train = data_y_train + [label] * int(samplesPerGesture * ratio)
    data_y_test = data_y_test + [label] * int(samplesPerGesture * (1 - ratio) + 1)
    #data_x_train = np.concatenate((data_x_train,data_x[int(label*samplesPerGesture):int(((label*samplesPerGesture) + samplesPerGesture/2 - 1))]),axis=0)
    print("Number: ", label, "is assigned to", gestureOrder[label])
  data_y = np.asarray(data_y)
  data_y_train = np.asarray(data_y_train)
  data_y_test = np.asarray(data_y_test)
  print(data_y_train)
  EVALUATION_INTERVAL = 200
  EPOCHS = 10
  BATCH_SIZE = 256
  BUFFER_SIZE = 10000


  # data_y_train =  tf.keras.utils.normalize(data_y_train)
  # print(data_y_train.shape)
  # data_y_test =  tf.keras.utils.normalize(data_y_test)
  # data_x_train =  tf.keras.utils.normalize(data_x_train)
  # data_x_test =  tf.keras.utils.normalize(data_x_test)
  train_data_single = tf.data.Dataset.from_tensor_slices((data_x_train, data_y_train))
  train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  print(">",data_x_test.shape)
  print(data_y_test.shape)
  val_data_single = tf.data.Dataset.from_tensor_slices((data_x_test, data_y_test))
  val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

  single_step_model = tf.keras.models.Sequential()
  print(data_x_train.shape[-2:])
  single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=data_x_train.shape[-2:]))

  data_y_train = tf.keras.utils.to_categorical(data_y_train)
  data_y_test = tf.keras.utils.to_categorical(data_y_test)
  print(data_y_train.shape[1])
  #single_step_model.add(tf.keras.layers.Dense(8, input_dim=4, activation='relu'))
  #single_step_model.add(Dropout(0.5))
  # print(">", data_y_train.shape[0])
  single_step_model.add(tf.keras.layers.Dense(100,activation='relu'))
  single_step_model.add(tf.keras.layers.Dense(2,activation = 'softmax'))
  single_step_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  single_step_model.fit(data_x_train, data_y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
  _, accuracy = single_step_model.evaluate(data_x_test, data_y_test, batch_size=BATCH_SIZE, verbose=0)
  print(data_x_test.shape)
  #single_step_model.
  print(single_step_model.predict_classes(data_x_test[:1]))
  #single_step_model.add(tf.keras.layers.Dense(6,activation="softmax"))
  #single_step_model.add(tf.keras.layers.Dense(1))

  # single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
  # #single_step_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # #single_step_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # for x, y in val_data_single.take(3):
  #   print(single_step_model.predict(x).shape)
  #   print(single_step_model.predict(x)[0])
  # single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
  #                                           steps_per_epoch=EVALUATION_INTERVAL,
  #                                           validation_data=val_data_single,
  #                                           validation_steps=50)
  # #print(val_data_single.shape)
  # for x, y in val_data_single.take(3):
  #   print(single_step_model.predict(x)[0])
  #   print(single_step_model.predict(x)[0].shape)
    #print(x)
    #print(single_step_model.predict_classes(x))

  # print(data_x_test.shape)
  # print(data_x_train.shape)
  # print(data_x_train[7][0][:])
  # print(data_y_train.shape)
  # print(data_y_test.shape)
  # print(data_x_test[0][0][:])
  #data_x_train = np.delete(data_x_train,[0],axis=0)
  #print(data_x_train.shape)
  #spittling data_x into different
  #data_x_train = data_x[0:int(samplesPerGesture/2)]
  #from here start create the model
