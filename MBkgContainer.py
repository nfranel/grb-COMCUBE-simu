# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports

# Developped modules imports
from funcmod import *


class BkgContainer:
  """
  Class containing the information for 1 background file
  """

  def __init__(self, datafile, sim_duration, save_pos, save_time, ergcut):
    """
    -data_list : 1 background tra file (unpol) from which extract the data
    """
    # Extraction of the background position (the important impormation is mostly its dec)
    self.dec, self.ra = datafile.split("_")[-2:]
    self.dec = float(self.dec)
    self.ra = float(self.ra.split(".inc")[0])
    # Attributes filled with file reading (or to be used from this moment)
    self.compton_ener = []
    self.compton_second = []
    self.single_ener = []
    if save_pos:
      self.compton_firstpos = []
      self.compton_secpos = []
      self.single_pos = []
    else:
      compton_firstpos = []
      compton_secpos = []
      single_pos = []
    if save_time:
      self.compton_time = []
      self.single_time = []
    self.single = 0
    self.single_cr = 0
    self.compton = 0
    self.compton_cr = 0
    # Attributes that are used while determining the detector where the interaction occured
    # self.calor = 0
    # self.dsssd = 0
    # self.side = 0
    # = []

    data_pol = readfile(datafile)
    for event in data_pol:
      reading = readevt(event, ergcut)
      # print("reading\n", reading)
      if len(reading) == 5:
        self.compton_second.append(reading[0])
        self.compton_ener.append(reading[1])
        if save_time:
          self.compton_time.append(reading[2])
        if save_pos:
          self.compton_firstpos.append(reading[3])
          self.compton_secpos.append(reading[4])
        else:
          compton_firstpos.append(reading[3])
          compton_secpos.append(reading[4])
      elif len(reading) == 3:
        self.single_ener.append(reading[0])
        if save_time:
          self.single_time.append(reading[1])
        if save_pos:
          self.single_pos.append(reading[2])
        else:
          single_pos.append(reading[2])
    self.compton_ener = np.array(self.compton_ener)
    self.compton_second = np.array(self.compton_second)
    self.single_ener = np.array(self.single_ener)
    if save_pos:
      self.compton_firstpos = np.array(self.compton_firstpos)
      self.compton_secpos = np.array(self.compton_secpos)
      self.single_pos = np.array(self.single_pos)
    else:
      compton_firstpos = np.array(compton_firstpos)
      compton_secpos = np.array(compton_secpos)
      single_pos = np.array(single_pos)

    if save_time:
      self.compton_time = np.array(self.compton_time)
      self.single_time = np.array(self.single_time)
    self.single = len(self.single_ener)
    self.single_cr = self.single / sim_duration
    self.compton = len(self.compton_ener)
    self.compton_cr = self.compton / sim_duration
