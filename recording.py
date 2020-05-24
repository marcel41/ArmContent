import open_myo as myo
import time
#global variable
isReadyToRegisterData = False
samplesPerSeconds = 0
#def section
#-------------------------------------------------------------------------------
def process_emg(emg):
  #global samplesPerSeconds
  if(isReadyToRegisterData):
    print("readings -> ", emg)
    global samplesPerSeconds
    samplesPerSeconds += 1
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  #global samplesPerSeconds
  myo_mac_addr = myo.get_myo()
  myo_device = myo.Device()
  myo_device.services.sleep_mode(1)  # never sleep
  myo_device.services.vibrate(1) #short vibration
  myo_device.services.emg_filt_notifications()
  print("Battery: %d" % myo_device.services.battery())
  myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)


  iterations = int(input("insert the number of iterations: "))
  timeHoldingGesture = int(input("how many seconds will you  hold the gesture: "))
  samplesPerGesture = int(input("insert number of samplesPerGesture"))
  time.sleep(2)
  myo_device.add_emg_event_handler(process_emg)
  for i in range(iterations):
    starting_time = time.time()
    isReadyToRegisterData = True
    while(round(time.time() - starting_time,1) <= timeHoldingGesture):
      if myo_device.services.waitForNotifications(1):
        continue #return to the beggining of while loop
      else:
        print("no data has been received from the peripheral, waiting...")
    isReadyToRegisterData = False
    print("total number of samples: ", samplesPerSeconds)
    samplesPerSeconds = 0;
    time.sleep(2)
