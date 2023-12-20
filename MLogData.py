# Autor Nathan Franel
# Date 06/12/2023
# Version 2 :
#

# Package imports
import numpy as np
# Developped modules imports
from catalog import Catalog


############################################################
# Usefull functions :
############################################################
class LogData:
  """
  Class containing the data from a log file
  """

  def __init__(self, sim_directory):
    self.sim_directory = sim_directory
    self.keys_description = None
    self.keys = None
    self.name = []
    self.sim_num = []
    self.sat_num = []
    self.status = []
    self.inc = []
    self.ohm = []
    self.omega = []
    self.alt = []
    self.sat_decwf = []
    self.sat_rawf = []
    self.grb_decwf = []
    self.grb_rawf = []
    self.grb_decsf = []
    self.grb_rasf = []

    self.extract_log()

  def extract_log(self):
    """
    Extracts the information from the log file
    """
    with open(self.sim_directory, "r") as f:
      lines = f.read().split("\n")
    self.keys_description = lines[2]
    self.keys = lines[3]
    for line in lines[6:]:
      data = line.split(" | ")
      self.name.append(data[0])
      self.sim_num.append(data[1])
      self.sat_num.append(data[2])
      self.status.append(data[3])
      self.inc.append(data[4])
      self.ohm.append(data[5])
      self.omega.append(data[6])
      self.alt.append(data[7])
      self.sat_decwf.append(data[8])
      self.sat_rawf.append(data[9])
      self.grb_decwf.append(data[10])
      self.grb_rawf.append(data[11])
      self.grb_decsf.append(data[12])
      self.grb_rasf.append(data[13])

  def detection_statistics(self):
    """
    Prints the detection statistics for a set a simulation
    """
    simulated = np.sum(np.where(self.status == "Simulated", 1, 0))
    horizon = np.sum(np.where(self.status == "Ignored(horizon)", 1, 0))
    off = np.sum(np.where(self.status == "Ignored(off)", 1, 0))
    print("The detection statistics for the simulated grbs is the following :")
    print(f"   Number of simulation possible : {len(self.name)}")
    print(f"   Number of simulation done : {simulated}")
    print(f"   Number of ignored simulation : {len(self.name) - simulated}")
    print(f"       With {horizon} ignored because the source is bellow the atmosphere")
    print(f"       With {off} ignored because the satellite if off")
    return simulated, horizon, off


# lgrb_cat = "./GBM/longGBM.txt"
# sgrb_cat = "./GBM/shortGBM.txt"
#
# cat = Catalog(lgrb_cat, [4, '\n', 5, '|', 2000])
# cat.spectral_information()
#
# log = LogData("/pdisk/ESA/test--400km--0-0-0--27sat")
# log.detection_statistics()
