# Autor Nathan Franel
# Date 06/12/2023
# Version 2 :
#

# Package imports
import numpy as np


# Developped modules imports


############################################################
# Usefull functions :
############################################################
class LogData:
  """
  Class containing the data from a log file
  """
  def __init__(self, sim_directory):
    """
    :param sim_directory: Directory where the simulation data is saved
    """
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
    self.rand_time = []
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
    logfile = f"{self.sim_directory}/simulation_logs.txt"
    with open(logfile, "r") as f:
      lines = f.read().split("\n")
    self.keys_description = lines[2]
    self.keys = lines[3]
    for line in lines[6:-1]:
      data = line.split(" | ")
      self.name.append(data[0])
      self.sim_num.append(int(data[1]))
      self.sat_num.append(int(data[2]))
      self.status.append(data[3])
      self.inc.append(float(data[4]))
      self.ohm.append(float(data[5]))
      self.omega.append(float(data[6]))
      self.alt.append(float(data[7]))
      self.rand_time.append(float(data[8]))
      self.sat_decwf.append(float(data[9]))
      self.sat_rawf.append(float(data[10]))
      self.grb_decwf.append(float(data[11]))
      self.grb_rawf.append(float(data[12]))
      if data[3] == 'Ignored(off)':
        self.grb_decsf.append(data[13])
        self.grb_rasf.append(data[14])
      else:
        self.grb_decsf.append(float(data[13]))
        self.grb_rasf.append(float(data[14]))
    # Changing the lists into np arrays
    self.name = np.array(self.name)
    self.sim_num = np.array(self.sim_num)
    self.sat_num = np.array(self.sat_num)
    self.status = np.array(self.status)
    self.inc = np.array(self.inc)
    self.ohm = np.array(self.ohm)
    self.omega = np.array(self.omega)
    self.alt = np.array(self.alt)
    self.rand_time = np.array(self.rand_time)
    self.sat_decwf = np.array(self.sat_decwf)
    self.sat_rawf = np.array(self.sat_rawf)
    self.grb_decwf = np.array(self.grb_decwf)
    self.grb_rawf = np.array(self.grb_rawf)
    self.grb_decsf = np.array(self.grb_decsf)
    self.grb_rasf = np.array(self.grb_rasf)

  def detection_statistics(self, cat):
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
    print(f"       With {off} ignored because the satellite is switch off")
    ret_name, ret_name_ite, ret_sim_ite, ret_sat_ite, ret_suffix_ite = self.detected_iteration_values(cat)
    return simulated, horizon, off, ret_name, ret_name_ite, ret_sim_ite, ret_sat_ite, ret_suffix_ite

  def detected_iteration_values(self, cat):
    name_temp = self.name[0]
    ite_grb = 0
    ret_name = []
    ret_name_ite = []
    ret_sim_ite = []
    ret_sat_ite = []
    ret_suffix_ite = []
    for ite, name in enumerate(self.name):
      if name_temp != name:
        ite_grb += 1
        name_temp = name
      if ite_grb >= len(cat.df.name):
        break
      if self.status[ite] == "Simulated":
        ret_name.append(name)
        ret_name_ite.append(ite_grb)
        ret_sim_ite.append(self.sim_num[ite])
        ret_sat_ite.append(self.sat_num[ite])
        ret_suffix_ite.append(f"{self.grb_decwf[ite]:.1f}_{self.grb_rawf[ite]:.1f}_{self.rand_time[ite]:.4f}")
    return ret_name, ret_name_ite, ret_sim_ite, ret_sat_ite, ret_suffix_ite

