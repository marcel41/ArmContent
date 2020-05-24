import open_myo as myo

#def section
#-------------------------------------------------------------------------------
def process_emg(emg):
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
  while True:
    if myo_device.services.waitForNotifications(1):
      continue #return to the beggining of while loop
    print("no data has been received from the peripheral, waiting...")
