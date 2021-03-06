import numpy as np
import tensorflow as tf
import open_myo as myo
from kulka import Kulka
import time
from multiprocessing import Process, Value
isReadyToRegisterData = False
samplesPerSeconds = 0
dataRecollectedPerIteration = list()

ADDR = '68:86:e7:00:ef:40'
#interpreter = tf.lite.Interpreter(model_path="myLittleModel.tflite")
interpreter = tf.lite.Interpreter(model_path="model2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

STEPS = 10
SPEED = 0x30
SLEEP_TIME = 0.3
labelPred = Value('i',0)

            #def section
#-------------------------------------------------------------------------------
def make_a_step(kulka, current_angle):
  kulka.roll(SPEED, current_angle)
  time.sleep(SLEEP_TIME)
  kulka.roll(0, current_angle)

def make_a_circle(kulka, steps):
  rotate_by = 360 // steps
  current_angle = 1

  for _ in range(steps):
    make_a_step(kulka, current_angle % 360)
    current_angle += rotate_by
def make_it_dance(kulka):
  speed = 0x88
  sleep_time = 0.3

  for angle in [1, 90, 180, 270]:
      kulka.roll(speed, angle)
      time.sleep(sleep_time)
def make_it_go_90(kulka):
  sleep_time = 1

  kulka.roll(SPEED, 90)
  time.sleep(sleep_time)
def make_it_go_1(kulka):
  sleep_time = 1

  kulka.roll(SPEED, 1)
  time.sleep(sleep_time)
def make_it_go_180(kulka):
  sleep_time = 1

  kulka.roll(SPEED, 180)
  time.sleep(sleep_time)
def make_it_go_270(kulka):
  sleep_time = 1

  kulka.roll(SPEED, 270)
  time.sleep(sleep_time)

  kulka.roll(0, 0)
def process_emg(emg):
  if(isReadyToRegisterData):
    #print("readings-> ", emg)
    global dataRecollectedPerIteration
    #dataRecollectedPerIteration += emg
    dataRecollectedPerIteration.append(emg)
    global samplesPerSeconds
    samplesPerSeconds += 1

def classifySignal():
  global interpreter
  global input_details
  global output_details
  global labelPred
  arrayXD = np.empty((1,150,8),dtype="float32")
  super_inp = np.asarray(dataRecollectedPerIteration,dtype="float32")
  arrayXD[0] = super_inp
  interpreter.set_tensor(input_details[0]['index'], arrayXD)
  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  labelPred.value = np.argmax(output_data)
  # if(np.argmax(output_data) == 1):
  #     make_a_circle(kulka, STEPS)
  # else:
  #   print("nothing")


def func1(labelPredXD):
  #print("HELP")
  with Kulka(ADDR) as kulka:
    while(True):
      #print(labelPredXD.value)
      if(labelPredXD.value == 3):
        print ('Do something')
        make_it_go_1(kulka)
      elif(labelPredXD.value == 2):
        print ('spread')
        make_it_go_180(kulka)
      elif(labelPredXD.value == 1):
        print ('dance')
        make_it_dance(kulka)
      else:
        print ('Do nothing')

def readAndClassify(samplesPerGesture):
  global isReadyToRegisterData
  global samplesPerSeconds
  global dataRecollectedPerIteration
  myo_mac_addr = myo.get_myo()
  myo_device = myo.Device()
  myo_device.services.sleep_mode(1)  # never sleep
  myo_device.services.vibrate(1) #short vibration
  myo_device.services.emg_filt_notifications()
  print("Battery: %d" % myo_device.services.battery())
  myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
  myo_device.add_emg_event_handler(process_emg)
  while(True):
    myo_device.services.vibrate(1) # short vibration to let user know we are recording
    #time.sleep(2) #add some delay to avoid the vibration causing any interference
    isReadyToRegisterData = True
    while(samplesPerSeconds < samplesPerGesture):
      if myo_device.services.waitForNotifications(1):
        continue #return to the beggining of while loop
      else:
        print("no data has been received from the peripheral, waiting...")
    isReadyToRegisterData = False
    print("received Info")
    classifySignal()
    dataRecollectedPerIteration.clear()
    print("total number of samples: ", samplesPerSeconds)
    samplesPerSeconds = 0;
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  myo_mac_addr = myo.get_myo()
  myo_device = myo.Device()
  myo_device.services.sleep_mode(1)  # never sleep
  myo_device.services.vibrate(1) #short vibration
  myo_device.services.emg_filt_notifications()
  print("Battery: %d" % myo_device.services.battery())



  myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
  samplesPerGesture = int(input("insert number of samplesPerGesture: "))
  myo_device.add_emg_event_handler(process_emg)

  p1 = Process(target=func1,args=(labelPred,))
  p1.start()
  # while(True):
  #   if(labelPred == 1):
  #     #print("Do something")
  #     pass
  #   elif(labelPred == 0):
  #     pass
      #print("do nothing")
  # with Kulka(ADDR) as kulka:
  while(True):
    myo_device.services.vibrate(1) # short vibration to let user know we are recording
    #time.sleep(2) #add some delay to avoid the vibration causing any interference
    isReadyToRegisterData = True
    while(samplesPerSeconds < samplesPerGesture):
      if myo_device.services.waitForNotifications(1):
            continue #return to the beggining of while loop
      else:
        print("no data has been received from the peripheral, waiting...")
    isReadyToRegisterData = False
    print("received Info")
    classifySignal()
    dataRecollectedPerIteration.clear()
    print("total number of samples: ", samplesPerSeconds)
    samplesPerSeconds = 0;
