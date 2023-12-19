# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import os
# Developped modules imports
from funcmod import *
from launch_bkg_sim import read_bkgpar


class BkgContainer(list):
  """
  Class containing the information for 1 background file
  """

  def __init__(self, bkgparfile, save_pos, save_time, ergcut):
    """
    :param bkgparfile : background parameter file
    :param save_pos : True if the interaction positions are to be saved
    :param save_time : True if the interaction times are to be saved
    :param ergcut : energy cut to apply
    """
    geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes = read_bkgpar(bkgparfile)
    self.geometry = geom       # To compare with data/mu100 and make sure everything works with the same softs
    self.revanfile = revanf    # To compare with data/mu100 and make sure everything works with the same softs
    self.mimrecfile = mimrecf  # To compare with data/mu100 and make sure everything works with the same softs
    self.sim_time = simtime
    self.lat_range = latitudes
    self.alt_range = altitudes

    geom_name = geom.split(".geo.setup")[0].split("/")[-1]
    saving = f"bkgsaved_{geom_name}_{np.min(self.lat_range):.0f}-{np.max(self.lat_range):.0f}-{len(self.lat_range):.0f}_{np.min(self.alt_range):.0f}-{np.max(self.alt_range):.0f}-{len(self.alt_range):.0f}.txt"

    if not saving in os.listdir(f"./bkg/sim_{geom_name}"):
      init_time = time()
      print("###########################################################################")
      print(" bkg data not saved : Saving ")
      print("###########################################################################")
      self.save_data(f"./bkg/sim_{geom_name}/{saving}")
      print("=======================================")
      print(" Saving of bkg data finished in : ", time() - init_time, "seconds")
      print("=======================================")
    init_time = time()
    print("###########################################################################")
    print(" Extraction of bkg data ")
    print("###########################################################################")
    list.__init__(self, self.read_data(f"./bkg/sim_{geom_name}/{saving}", save_pos, save_time, ergcut))
    print("=======================================")
    print(" Extraction of bkg data finished in : ", time() - init_time, "seconds")
    print("=======================================")

  def save_data(self, file):
    """
    Function used to save the bkg data into a txt file
    """
    with open(file, "w") as f:
      f.write("# File containing background data for : \n")
      f.write(f"# Geometry : {self.geometry}\n")
      f.write(f"# Revan file : {self.revanfile}\n")
      f.write(f"# Mimrec file : {self.mimrecfile}\n")
      f.write(f"# Simulation time : {self.sim_time}\n")
      f.write(f"# Latitude min-max-number of value : {np.min(self.lat_range)}-{np.max(self.lat_range)}-{len(self.lat_range)}\n")
      f.write(f"# Altitude list : {self.alt_range}\n")
      f.write("# Keys : dec | compton_ener | compton_second | single_ener | compton_firstpos | compton_secpos | single_pos | compton_time | single_time | single | single_cr | compton | compton_cr | calor | dsssd | side\n")
      for alt in self.alt_range:
        for lat in self.lat_range:
          geom_name = self.geometry.split(".geo.setup")[0].split("/")[-1]
          data = readfile(f"./bkg/sim_{geom_name}/sim/bkg_{alt:.1f}_{lat:.1f}_{self.sim_time:.0f}s.inc1.id1.extracted.tra")
          decbkg = 90 - lat
          altbkg = alt
          compton_second = []
          compton_ener = []
          compton_time = []
          compton_firstpos = []
          compton_secpos = []
          single_ener = []
          single_time = []
          single_pos = []
          for event in data:
            reading = readevt(event, None)
            if len(reading) == 5:
              compton_second.append(reading[0])
              compton_ener.append(reading[1])
              compton_time.append(reading[2])
              compton_firstpos.append(reading[3])
              compton_secpos.append(reading[4])
            elif len(reading) == 3:
              single_ener.append(reading[0])
              single_time.append(reading[1])
              single_pos.append(reading[2])
          f.write("NewBkg\n")
          f.write(f"{decbkg}\n")
          f.write(f"{altbkg}\n")
          for ite in range(len(compton_second) - 1):
            f.write(f"{compton_second[ite]}|")
          f.write(f"{compton_second[-1]}\n")
          for ite in range(len(compton_ener) - 1):
            f.write(f"{compton_ener[ite]}|")
          f.write(f"{compton_ener[-1]}\n")
          for ite in range(len(compton_time) - 1):
            f.write(f"{compton_time[ite]}|")
          f.write(f"{compton_time[-1]}\n")
          for ite in range(len(compton_firstpos) - 1):
            string = f"{compton_firstpos[ite][0]}_{compton_firstpos[ite][1]}_{compton_firstpos[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_firstpos[-1][0]}_{compton_firstpos[-1][1]}_{compton_firstpos[-1][2]}"
          f.write(f"{string}\n")
          for ite in range(len(compton_secpos) - 1):
            string = f"{compton_secpos[ite][0]}_{compton_secpos[ite][1]}_{compton_secpos[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_secpos[-1][0]}_{compton_secpos[-1][1]}_{compton_secpos[-1][2]}"
          f.write(f"{string}\n")
          for ite in range(len(single_ener) - 1):
            f.write(f"{single_ener[ite]}|")
          f.write(f"{single_ener[-1]}\n")
          for ite in range(len(single_time) - 1):
            f.write(f"{single_time[ite]}|")
          f.write(f"{single_time[-1]}\n")
          for ite in range(len(single_pos) - 1):
            string = f"{single_pos[ite][0]}_{single_pos[ite][1]}_{single_pos[ite][2]}"
            f.write(f"{string}|")
          string = f"{single_pos[-1][0]}_{single_pos[-1][1]}_{single_pos[-1][2]}"
          f.write(f"{string}\n")

  def read_data(self, file, save_pos, save_time, ergcut):
    """
    Function used to read the bkg txt file
    """
    with open(file, "r") as f:
      files_saved = f.read().split("NewBkg\n")
    return [BkgData(file_saved, self.sim_time, save_pos, save_time, ergcut) for file_saved in files_saved[1:]]


class BkgData:
  """
  Class containing the information for 1 background file
  """

  def __init__(self, data, sim_duration, save_pos, save_time, ergcut):
    """
    :param data : str containing all the data from a background file
    :param sim_duration : duration of the background simulation
    :param save_pos : True if the interaction positions are to be saved
    :param save_time : True if the interaction times are to be saved
    :param ergcut : energy cut to apply
    """
    # Extraction of the background values
    lines = data.split("\n")
    self.dec = float(lines[0])
    self.alt = float(lines[1])
    # Attributes filled with file reading (or to be used from this moment)
    self.compton_second = np.array(lines[2].split("|"), dtype=float)
    self.compton_ener = np.array(lines[3].split("|"), dtype=float)
    self.single_ener = np.array(lines[7].split("|"), dtype=float)
    if save_time:
      self.compton_time = np.array(lines[4].split("|"), dtype=float)
      self.single_time = np.array(lines[8].split("|"), dtype=float)
    else:
      self.compton_time = None
      self.single_time = None
    if save_pos:
      self.compton_firstpos = np.array([val.split("_") for val in lines[5].split("|")], dtype=float)
      self.compton_secpos = np.array([val.split("_") for val in lines[6].split("|")], dtype=float)
      self.single_pos = np.array([val.split("_") for val in lines[9].split("|")], dtype=float)
    else:
      self.compton_firstpos = None
      self.compton_secpos = None
      self.single_pos = None
    if ergcut is not None:
      compton_index = np.where(self.compton_ener >= ergcut[0], np.where(self.compton_ener <= ergcut[1], True, False), False)
      single_index = np.where(self.single_ener >= ergcut[0], np.where(self.single_ener <= ergcut[1], True, False), False)
      self.compton_second = self.compton_second[compton_index]
      self.compton_ener = self.compton_ener[compton_index]
      self.single_ener = self.single_ener[single_index]
      if save_time:
        self.compton_time = self.compton_time[compton_index]
        self.single_time = self.single_time[single_index]
      if save_pos:
        self.compton_firstpos = self.compton_firstpos[compton_index]
        self.compton_secpos = self.compton_secpos[compton_index]
        self.single_pos = self.single_pos[single_index]

    self.single = len(self.single_ener)
    self.single_cr = self.single / sim_duration
    self.compton = len(self.compton_ener)
    self.compton_cr = self.compton / sim_duration
