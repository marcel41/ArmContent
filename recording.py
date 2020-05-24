import open_myo as myo

#def section
#-------------------------------------------------------------------------------
def process_emg(emg):
  print("readings -> ", emg)
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  myo_mac_addr = myo.get_myo()
  myo_device = myo.Device()
  myo_device.services.sleep_mode(0)
