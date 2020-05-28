import open_myo as myo
import time
#global variable
isReadyToRegisterData = False
samplesPerSeconds = 0
dataRecollectedPerIteration = list()
#def section
#-------------------------------------------------------------------------------
def process_emg(emg):
  if(isReadyToRegisterData):
    print("readings-> ", emg)
    global dataRecollectedPerIteration
    dataRecollectedPerIteration += emg
    global samplesPerSeconds
    samplesPerSeconds += 1

def saving_recording():
  f.write(str(dataRecollectedPerIteration))
  f.write("\n")
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  myo_mac_addr = myo.get_myo()
  myo_device = myo.Device()
  myo_device.services.sleep_mode(1)  # never sleep
  myo_device.services.vibrate(1) #short vibration
  myo_device.services.emg_filt_notifications()
  print("Battery: %d" % myo_device.services.battery())



  myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
  iterations = int(input("insert the number of iterations: "))
  samplesPerGesture = int(input("insert number of samplesPerGesture: "))
  nameOfGesture = input("insert name of the the gesture:")
  f = open("dataSet/" + nameOfGesture + ".txt","w")
  myo_device.services.vibrate(1) # short vibration to let user know we are recording
  time.sleep(2) #add some delay to avoid the vibration causing any interference
  myo_device.add_emg_event_handler(process_emg)
  for i in range(iterations):
    isReadyToRegisterData = True
    while(samplesPerSeconds < samplesPerGesture):
      if myo_device.services.waitForNotifications(1):
        continue #return to the beggining of while loop
      else:
        print("no data has been received from the peripheral, waiting...")
    isReadyToRegisterData = False
    saving_recording()
    dataRecollectedPerIteration.clear()
    print("total number of samples: ", samplesPerSeconds)
    samplesPerSeconds = 0;
    myo_device.services.vibrate(1) # short vibration to let user know we are recording
    time.sleep(2) #add some delay to avoid the vibration causing any interference
  print (dataRecollectedPerIteration)
  #saving_recording()
