import tensorflow as tf

import numpy as np
import glob
import io
import os
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
  data_y = None
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

      y_label_Class = os.path.basename(fd.name)
      if(firstSample):
        data_x = np.copy(np_examples)
        firstSample = False
      else:
        data_x = np.concatenate((data_x,np_examples),axis=0)
      gestureOrder.append(y_label_Class)
      #value_label = [len(gestureOrder)] * 10
      #data_y.append(value_label)
      myList.clear()
      print(data_x.shape)
      print(data_x[9][1][:])
  print(data_x[9][1][:])
  print(data_x[19][1][:])
  print(data_x[29][1][:])
  print(data_x[39][1][:])
  samplesPerGesture = int(len(data_x)/len(gestureOrder))
  data_y = list()
  for label in range(len(gestureOrder)):
    data_y = data_y + ([label] * samplesPerGesture)
    print("Number: ", label, "is assigned to", gestureOrder[label])
  data_y = np.asarray(data_y)
  print(data_y.shape)
  print(data_y[9])
  print(data_y[19])
  print(data_y[29])
  print(data_y[39])
