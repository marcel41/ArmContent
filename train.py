import tensorflow as tf

import numpy as np
import os

ftest = open("dataSet/fist.txt","r")
myList = list()
np_array_reshaped = None
#np_examples = numpy.zeros(f)
count = 0
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
ftest.close()
