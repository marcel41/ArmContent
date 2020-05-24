import open_myo as myo
import time
#global variable
isReadyToRegisterData = False
#def section
#-------------------------------------------------------------------------------
def process_emg(emg):
  if(isReadyToRegisterData):
    print("readings -> ", emg)
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  myo_mac_addr = myo.get_myo()
  myo_device = myo.Device()
  myo_device.services.sleep_mode(1)  # never sleep
  myo_device.services.vibrate(1) #short vibration
  myo_device.services.emg_filt_notifications()
  print("Battery: %d" % myo_device.services.battery())
  myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
  myo_device.add_emg_event_handler(process_emg)

  iterations = int(input("insert the number of iterations"))
  timeHoldingGesture = int(input("how many seconds will you  hold the gesture"))
  for i in range(iterations):
    starting_time = time.time() - start
    isReadyToRegisterData = True
    while(round(time.time() - starting_time,1) <= timeHoldingGesture):
      if myo_device.services.waitForNotifications(1):
        continue #return to the beggining of while loop
      else:
        print("no data has been received from the peripheral, waiting...")
    isReadyToRegisterData = False
    time.sleep(2)
