# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports

# Developped modules imports
from funcmod import *
from launch_bkg_sim import read_bkgpar


class BkgContainer(list):
  """
  Class containing the information for 1 background file
  """
  def __init__(self, bkgparfile, save_time, ergcut, special_name=None):
    """
    :param bkgparfile: background parameter file
    :param save_time: True if the interaction times are to be saved
    :param ergcut: energy cut to apply
    """
    geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes = read_bkgpar(bkgparfile)
    self.geometry = geom       # TODO compare with data/mu100 and make sure everything works with the same softs
    self.revanfile = revanf    # compare with data/mu100 and make sure everything works with the same softs
    self.mimrecfile = mimrecf  # compare with data/mu100 and make sure everything works with the same softs
    self.sim_time = simtime
    self.lat_range = latitudes
    self.alt_range = altitudes
    if special_name is None:
      self.fold_name = geom.split(".geo.setup")[0].split("/")[-1]
    else:
      self.fold_name = special_name
    saving = f"bkgsaved_{self.fold_name}_{np.min(self.lat_range):.0f}-{np.max(self.lat_range):.0f}-{len(self.lat_range):.0f}_{np.min(self.alt_range):.0f}-{np.max(self.alt_range):.0f}-{len(self.alt_range):.0f}.txt"
    cond_saving = f"cond_bkgsaved_{self.fold_name}_{np.min(self.lat_range):.0f}-{np.max(self.lat_range):.0f}-{len(self.lat_range):.0f}_{np.min(self.alt_range):.0f}-{np.max(self.alt_range):.0f}-{len(self.alt_range):.0f}_ergcut-{ergcut[0]}-{ergcut[1]}.txt"
    # print("self.fold_name", self.fold_name)
    # print("saving", saving)
    # print("cond_saving", cond_saving)
    if cond_saving not in os.listdir(f"./bkg/sim_{self.fold_name}"):
      if saving not in os.listdir(f"./bkg/sim_{self.fold_name}"):
        init_time = time()
        print("###########################################################################")
        print(" bkg data not saved : Saving ")
        print("###########################################################################")
        self.save_data(f"./bkg/sim_{self.fold_name}/{saving}", f"./bkg/sim_{self.fold_name}/{cond_saving}", ergcut)
        print("=======================================")
        print(" Saving of bkg data finished in : ", time() - init_time, "seconds")
        print("=======================================")
      else:
        init_time = time()
        print("###########################################################################")
        print(" bkg condensed data not saved : Saving ")
        print("###########################################################################")
        self.save_cond_only(f"./bkg/sim_{self.fold_name}/{saving}", f"./bkg/sim_{self.fold_name}/{cond_saving}", ergcut)
        print("=======================================")
        print(" Saving of bkg data finished in : ", time() - init_time, "seconds")
        print("=======================================")

    init_time = time()
    print("###########################################################################")
    print(" Extraction of bkg data ")
    print("###########################################################################")
    # # Saving the data with a full format
    # list.__init__(self, self.read_data(f"./bkg/sim_{self.fold_name}/{saving}", save_time, ergcut, data_type="full"))
    # Saving the data with a condensed format
    list.__init__(self, self.read_data(f"./bkg/sim_{self.fold_name}/{cond_saving}", save_time, ergcut, data_type="cond"))
    print("=======================================")
    print(" Extraction of bkg data finished in : ", time() - init_time, "seconds")
    print("=======================================")

  def save_data(self, file, condensed_file, ergcut):
    """
    Function used to save the bkg data into a txt file
    :param file: path of the file containing saved data
    :param condensed_file: path of the file containing condensed saved data
    :param ergcut: energy cut used for making the condensed data file
    """
    with open(file, "w") as f:
      with open(condensed_file, "w") as fcond:
        f.write("# File containing background data for : \n")
        f.write(f"# Geometry : {self.geometry}\n")
        f.write(f"# Revan file : {self.revanfile}\n")
        f.write(f"# Mimrec file : {self.mimrecfile}\n")
        f.write(f"# Simulation time : {self.sim_time}\n")
        f.write(f"# Latitude min-max-number of value : {np.min(self.lat_range)}-{np.max(self.lat_range)}-{len(self.lat_range)}\n")
        f.write(f"# Altitude list : {self.alt_range}\n")
        f.write("# Keys : dec | alt | compton_second | compton_ener | compton_time | compton_firstpos | compton_secpos | single_ener | single_time | single_pos\n")

        fcond.write("# File containing condensed background data for : \n")
        fcond.write(f"# Geometry : {self.geometry}\n")
        fcond.write(f"# Revan file : {self.revanfile}\n")
        fcond.write(f"# Mimrec file : {self.mimrecfile}\n")
        fcond.write(f"# Simulation time : {self.sim_time}\n")
        fcond.write(f"# Latitude min-max-number of value : {np.min(self.lat_range)}-{np.max(self.lat_range)}-{len(self.lat_range)}\n")
        fcond.write(f"# Altitude list : {self.alt_range}\n")
        fcond.write("# Keys : dec | alt | compton_cr | single_cr | det_stat_compton | det_stat_single\n")
        for alt in self.alt_range:
          for lat in self.lat_range:
            # self.fold_name = self.geometry.split(".geo.setup")[0].split("/")[-1]
            data = readfile(f"./bkg/sim_{self.fold_name}/sim/bkg_{alt:.1f}_{lat:.1f}_{self.sim_time:.0f}s.inc1.id1.extracted.tra")
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
              if len(reading) == 9:
                compton_second.append(reading[0])
                compton_ener.append(reading[1])
                compton_time.append(reading[2])
                compton_firstpos.append(reading[3])
                compton_secpos.append(reading[4])
              elif len(reading) == 3:
                single_ener.append(reading[0])
                single_time.append(reading[1])
                single_pos.append(reading[2])

            compton_second = np.array(compton_second)
            compton_ener = np.array(compton_ener)
            compton_time = np.array(compton_time)
            compton_firstpos = np.array(compton_firstpos)
            compton_secpos = np.array(compton_secpos)
            single_ener = np.array(single_ener)
            single_time = np.array(single_time)
            single_pos = np.array(single_pos)

            f.write("NewBkg\n")
            save_value(f, decbkg)
            save_value(f, altbkg)
            save_value(f, compton_second)
            save_value(f, compton_ener)
            save_value(f, compton_time)
            if len(compton_ener) > 0:
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
            else:
              f.write(f"\n")
            save_value(f, single_ener)
            save_value(f, single_time)
            if len(single_ener) > 0:
              for ite in range(len(single_pos) - 1):
                string = f"{single_pos[ite][0]}_{single_pos[ite][1]}_{single_pos[ite][2]}"
                f.write(f"{string}|")
              string = f"{single_pos[-1][0]}_{single_pos[-1][1]}_{single_pos[-1][2]}"
              f.write(f"{string}\n")
            else:
              f.write(f"\n")

            compton_index = np.where(compton_ener >= ergcut[0], np.where(compton_ener <= ergcut[1], True, False), False)
            single_index = np.where(single_ener >= ergcut[0], np.where(single_ener <= ergcut[1], True, False), False)
            compton_ener = compton_ener[compton_index]
            single_ener = single_ener[single_index]

            # Détermination des détecteurs d'interaction
            compton_firstpos = compton_firstpos[compton_index]
            compton_secpos = compton_secpos[compton_index]
            single_pos = single_pos[single_index]
            compton_first_detector, compton_sec_detector, single_detector = find_detector(compton_firstpos, compton_secpos, single_pos, self.geometry)

            det_stat_compton = det_counter(np.concatenate((compton_first_detector, compton_sec_detector))).flatten()
            det_stat_single = det_counter(np.concatenate((compton_first_detector, compton_sec_detector))).flatten()
            # Writing the condensed file
            fcond.write("NewBkg\n")
            fcond.write(f"{decbkg}\n")
            fcond.write(f"{altbkg}\n")
            fcond.write(f"{len(compton_ener) / self.sim_time}\n")
            fcond.write(f"{len(single_ener) / self.sim_time}\n")
            save_value(fcond, det_stat_compton)
            save_value(fcond, det_stat_single)

  def read_data(self, file, save_time, ergcut, data_type="cond"):
    """
    Function used to read the bkg txt file
    :param file: path of the file containing the saved data (either full or condensed one)
    :param save_time: If True saves the interaction time when reading full data (data_type="full")
    :param ergcut: energy cut used for making the condensed data file
    :param data_type: Which file should be read
      If cond, condensed file is read
      If full, uncondensed file is read and save_time is used
    """
    with open(file, "r") as f:
      files_saved = f.read().split("NewBkg\n")
    return [BkgData(file_saved, self.sim_time, save_time, ergcut, self.geometry, data_type) for file_saved in files_saved[1:]]

  def save_cond_only(self, file, condensed_file, ergcut):
    """
    Function used to save the condensed bkg data into a txt file after extracting uncondensed data
    :param file: path for the full data file
    :param condensed_file: path for the condensed data file
    :param ergcut: energy window that should be applied on the data
    """
    with open(file, "r") as f:
      files_saved = f.read().split("NewBkg\n")
    with open(condensed_file, "w") as fcond:
      fcond.write("# File containing condensed background data for : \n")
      fcond.write(f"# Geometry : {self.geometry}\n")
      fcond.write(f"# Revan file : {self.revanfile}\n")
      fcond.write(f"# Mimrec file : {self.mimrecfile}\n")
      fcond.write(f"# Simulation time : {self.sim_time}\n")
      fcond.write(f"# Latitude min-max-number of value : {np.min(self.lat_range)}-{np.max(self.lat_range)}-{len(self.lat_range)}\n")
      fcond.write(f"# Altitude list : {self.alt_range}\n")
      fcond.write("# Keys : dec | alt | compton_cr | single_cr | det_stat_compton | det_stat_single\n")
      for file in files_saved[1:]:
        lines = file.split("\n")
        decbkg = float(lines[0])
        altbkg = float(lines[1])
        # Attributes filled with file reading (or to be used from this moment)
        # compton_second = np.fromstring(lines[2], sep='|', dtype=np.float32)
        compton_ener = np.fromstring(lines[3], sep='|', dtype=np.float32)
        # compton_time = np.fromstring(lines[4], sep='|', dtype=np.float32)
        compton_firstpos = np.array([val.split("_") for val in lines[5].split("|")], dtype=np.float32)
        compton_secpos = np.array([val.split("_") for val in lines[6].split("|")], dtype=np.float32)

        single_ener = np.fromstring(lines[7], sep='|', dtype=np.float32)
        # single_time = np.fromstring(lines[8], sep='|', dtype=np.float32)
        single_pos = np.array([val.split("_") for val in lines[9].split("|")], dtype=np.float32)

        # Applying the ergcut
        compton_index = np.where(compton_ener >= ergcut[0], np.where(compton_ener <= ergcut[1], True, False), False)
        single_index = np.where(single_ener >= ergcut[0], np.where(single_ener <= ergcut[1], True, False), False)
        compton_ener = compton_ener[compton_index]
        single_ener = single_ener[single_index]

        # Détermination des détecteurs d'interaction
        compton_firstpos = compton_firstpos[compton_index]
        compton_secpos = compton_secpos[compton_index]
        single_pos = single_pos[single_index]
        compton_first_detector, compton_sec_detector, single_detector = find_detector(compton_firstpos, compton_secpos, single_pos, self.geometry)

        det_stat_compton = det_counter(np.concatenate((compton_first_detector, compton_sec_detector))).flatten()
        det_stat_single = det_counter(np.concatenate((compton_first_detector, compton_sec_detector))).flatten()
        # Writing the condensed file
        fcond.write("NewBkg\n")
        fcond.write(f"{decbkg}\n")
        fcond.write(f"{altbkg}\n")
        fcond.write(f"{len(compton_ener) / self.sim_time}\n")
        fcond.write(f"{len(single_ener) / self.sim_time}\n")
        save_value(fcond, det_stat_compton)
        save_value(fcond, det_stat_single)


class BkgData:
  """
  Class containing the information for 1 background file
  """
  def __init__(self, data, sim_duration, save_time, ergcut, geometry, data_type="cond"):
    """
    :param data: str containing all the data from a background file
    :param sim_duration: duration of the background simulation
    :param save_time: True if the interaction times are to be saved
    :param ergcut: energy cut to apply
    :param geometry: geometry used for the simulation
    :param data_type: Which file should be read
      If cond, condensed file is read
      If full, uncondensed file is read and save_time is used
    """
    if data_type == "full":
      # Extraction of the background values
      lines = data.split("\n")
      self.dec = float(lines[0])
      # self.ra TODO
      self.alt = float(lines[1])
      # Attributes filled with file reading (or to be used from this moment)
      self.compton_second = np.fromstring(lines[2], sep='|', dtype=np.float32)
      self.compton_ener = np.fromstring(lines[3], sep='|', dtype=np.float32)
      self.single_ener = np.fromstring(lines[7], sep='|', dtype=np.float32)
      if save_time:
        self.compton_time = np.fromstring(lines[4], sep='|', dtype=np.float32)
        self.single_time = np.fromstring(lines[8], sep='|', dtype=np.float32)
      else:
        self.compton_time = None
        self.single_time = None
      compton_firstpos = np.array([val.split("_") for val in lines[5].split("|")], dtype=np.float32)
      compton_secpos = np.array([val.split("_") for val in lines[6].split("|")], dtype=np.float32)
      single_pos = np.array([val.split("_") for val in lines[9].split("|")], dtype=np.float32)
      if ergcut is not None:
        compton_index = np.where(self.compton_ener >= ergcut[0], np.where(self.compton_ener <= ergcut[1], True, False), False)
        single_index = np.where(self.single_ener >= ergcut[0], np.where(self.single_ener <= ergcut[1], True, False), False)
        self.compton_second = self.compton_second[compton_index]
        self.compton_ener = self.compton_ener[compton_index]
        self.single_ener = self.single_ener[single_index]
        if save_time:
          self.compton_time = self.compton_time[compton_index]
          self.single_time = self.single_time[single_index]
        compton_firstpos = compton_firstpos[compton_index]
        compton_secpos = compton_secpos[compton_index]
        single_pos = single_pos[single_index]

      self.single = len(self.single_ener)
      self.single_cr = self.single / sim_duration
      self.compton = len(self.compton_ener)
      self.compton_cr = self.compton / sim_duration

      self.compton_first_detector, self.compton_sec_detector, self.single_detector = find_detector(compton_firstpos, compton_secpos, single_pos, geometry)

    elif data_type == "cond":
      # Extraction of the background values
      lines = data.split("\n")
      self.dec = float(lines[0])
      self.alt = float(lines[1])
      self.compton_cr = float(lines[2])
      self.single_cr = float(lines[3])
      self.det_stat_compton = np.fromstring(lines[4], sep='|', dtype=np.int64)
      self.det_stat_single = np.fromstring(lines[5], sep='|', dtype=np.int64)

      # Attributes not filled because only the condensed data are extracted
      self.single = None
      self.compton = None
      self.compton_second = None
      self.compton_ener = None
      self.single_ener = None
      self.compton_time = None
      self.single_time = None

    else:
      print("Extraction impossible, wrong data_type given, only 'cond' and 'full' are possible")
