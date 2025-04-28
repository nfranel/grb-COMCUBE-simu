# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import numpy as np
from time import time
import os

import pandas as pd

# Developped modules imports
from src.General.funcmod import readevt, readfile, save_value, det_counter, find_detector, analyze_bkg_event
from src.Launchers.launch_bkg_sim import read_bkgpar


class BkgContainer:
  """
  Class containing the information for 1 background file
  """
  def __init__(self, bkgparfile, save_time, ergcut, special_name=None):
    """
    :param bkgparfile: background parameter file
    :param save_time: True if the interaction times are to be saved
    :param ergcut: energy cut to apply
    """
    self.array_dtype = np.float32
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
    saving = f"bkgsaved_{self.fold_name}_{np.min(self.lat_range):.0f}-{np.max(self.lat_range):.0f}-{len(self.lat_range):.0f}_{np.min(self.alt_range):.0f}-{np.max(self.alt_range):.0f}-{len(self.alt_range):.0f}.h5"
    cond_saving = f"cond_bkg-saved_{self.fold_name}_{np.min(self.lat_range):.0f}-{np.max(self.lat_range):.0f}-{len(self.lat_range):.0f}_{np.min(self.alt_range):.0f}-{np.max(self.alt_range):.0f}-{len(self.alt_range):.0f}_ergcut-{ergcut[0]}-{ergcut[1]}.txt"
    if cond_saving not in os.listdir(f"../Data/bkg/sim_{self.fold_name}"):
      if saving not in os.listdir(f"../Data/bkg/sim_{self.fold_name}"):
        init_time = time()
        print("###########################################################################")
        print(" bkg data not saved : Saving ")
        print("###########################################################################")
        self.save_fulldata(f"../Data/bkg/sim_{self.fold_name}/{saving}", f"../Data/bkg/sim_{self.fold_name}/{cond_saving}", ergcut)
        print("=======================================")
        print(" Saving of bkg data finished in : ", time() - init_time, "seconds")
        print("=======================================")
      else:
        init_time = time()
        print("###########################################################################")
        print(" bkg condensed data not saved : Saving ")
        print("###########################################################################")
        self.save_condensed_data(f"../Data/bkg/sim_{self.fold_name}/{saving}", f"../Data/bkg/sim_{self.fold_name}/{cond_saving}", ergcut)
        print("=======================================")
        print(" Saving of bkg data finished in : ", time() - init_time, "seconds")
        print("=======================================")

    init_time = time()
    print("###########################################################################")
    print(" Extraction of bkg data ")
    print("###########################################################################")
    # # Saving the data with a full format
    # list.__init__(self, self.read_data(f"../Data/bkg/sim_{self.fold_name}/{saving}", save_time, ergcut, data_type="full"))
    # Saving the data with a condensed format
    self.bkgdf = self.read_data(f"../Data/bkg/sim_{self.fold_name}/{cond_saving}", save_time, ergcut, data_type="cond")
    print("=======================================")
    print(" Extraction of bkg data finished in : ", time() - init_time, "seconds")
    print("=======================================")

  def save_fulldata(self, file, condensed_file, ergcut):
    """
    Function used to save the bkg data into a txt file
    :param file: path of the file containing saved data
    :param condensed_file: path of the file containing condensed saved data
    :param ergcut: energy cut used for making the condensed data file
    """
    with pd.HDFStore(file, mode="w") as f:
      f.get_storer("/").attrs.description = f"# File containing background data for : \n# Geometry : {self.geometry}\n# Revan file : {self.revanfile}\n# Mimrec file : {self.mimrecfile}\n# Simulation time : {self.sim_time}\n# Altitude list : {self.alt_range}"
      f.get_storer("/").attrs.structure = "Keys : bkgalt-bkgdec/compton or single dataframes"

      bkg_tab = []
      for alt in self.alt_range:
        for lat in self.lat_range:
          decbkg, altbkg, compton_second, compton_ener, compton_time, single_ener, single_time, compton_first_detector, compton_sec_detector, single_detector = analyze_bkg_event(f"../Data/bkg/sim_{self.fold_name}/sim/bkg_{alt:.1f}_{lat:.1f}_{self.sim_time:.0f}s.inc1.id1.extracted.tra", lat, alt, self.geometry, self.array_dtype)

          df_compton = pd.DataFrame({"compton_ener": compton_ener, "compton_second": compton_second, "compton_time": compton_time,
             "compton_first_detector": compton_first_detector, "compton_sec_detector": compton_sec_detector})
          df_single = pd.DataFrame({"single_ener": single_ener, "single_time": single_time, "single_detector": single_detector})
          key = f"{altbkg}-{decbkg}"
          # Saving Compton event related quantities
          f.put(f"{key}/compton", df_compton)
          # Saving single event related quantities
          f.put(f"{key}/single", df_single)
          # Saving scalar values
          # Specific to satellite
          f.get_storer(f"{key}/compton").attrs.decbkg = decbkg
          f.get_storer(f"{key}/compton").attrs.altbkg = altbkg

          df_compton = df_compton[(df_compton.compton_ener >= ergcut[0]) & (df_compton.compton_ener <= ergcut[1])]
          df_single = df_single[(df_single.single_ener >= ergcut[0]) & (df_single.single_ener <= ergcut[1])]
          det_stat_compton = det_counter(np.concatenate((df_compton.compton_first_detector.values, df_compton.compton_sec_detector.values))).flatten()
          det_stat_single = det_counter(df_single.single_detector.values).flatten()
          # Writing the condensed file
          bkg_tab.append([altbkg, decbkg, len(df_compton) / self.sim_time, len(df_single) / self.sim_time, det_stat_compton, det_stat_single])
    columns = ["bkg_alt", "bkg_dec", "compton_cr", "single_cr", "com_det_idx", "sin_det_idx"]
    cond_df = pd.DataFrame(data=bkg_tab, columns=columns)

    with pd.HDFStore(condensed_file, mode="w") as fcond:
      fcond.get_storer("/").attrs.description = f"# File containing CONDENSED background data for : \n# Geometry : {self.geometry}\n# Revan file : {self.revanfile}\n# Mimrec file : {self.mimrecfile}\n# Simulation time : {self.sim_time}\n# Altitude list : {self.alt_range}\n"
      fcond.get_storer("/").attrs.structure = "Keys : bkgalt-bkgdec/compton or single dataframes"
      fcond.get_storer("/").attrs.ergcut = f"energy cut : {ergcut[0]}-{ergcut[1]}"
      fcond.put(f"bkg_df", cond_df)

  def save_condensed_data(self, file, condensed_file, ergcut):
    """
    Function used to save the condensed bkg data into a txt file after extracting uncondensed data
    :param file: path for the full data file
    :param condensed_file: path for the condensed data file
    :param ergcut: energy window that should be applied on the data
    """
    bkg_tab = []
    with pd.HDFStore(file, mode="r") as f:
      for key in set(k.split("/")[1] for k in f.keys()):
        decbkg = f.get_storer(f"{key}/compton").attrs.decbkg
        altbkg = f.get_storer(f"{key}/compton").attrs.altbkg
        df_compton = f[f"{key}/compton"]
        df_single = f[f"{key}/single"]

        df_compton = df_compton[(df_compton.compton_ener >= ergcut[0]) & (df_compton.compton_ener <= ergcut[1])]
        df_single = df_single[(df_single.single_ener >= ergcut[0]) & (df_single.single_ener <= ergcut[1])]
        det_stat_compton = det_counter(np.concatenate((df_compton.compton_first_detector.values, df_compton.compton_sec_detector.values))).flatten()
        det_stat_single = det_counter(df_single.single_detector.values).flatten()
        # Writing the condensed file
        bkg_tab.append([altbkg, decbkg, len(df_compton) / self.sim_time, len(df_single) / self.sim_time, det_stat_compton, det_stat_single])

    columns = ["bkg_alt", "bkg_dec", "compton_cr", "single_cr", "com_det_idx", "sin_det_idx"]
    cond_df = pd.DataFrame(data=bkg_tab, columns=columns)

    with pd.HDFStore(condensed_file, mode="w") as fcond:
      fcond.get_storer("/").attrs.description = f"# File containing CONDENSED background data for : \n# Geometry : {self.geometry}\n# Revan file : {self.revanfile}\n# Mimrec file : {self.mimrecfile}\n# Simulation time : {self.sim_time}\n# Altitude list : {self.alt_range}\n"
      fcond.get_storer("/").attrs.structure = "Keys : bkg dataframe"
      fcond.get_storer("/").attrs.ergcut = f"energy cut : {ergcut[0]}-{ergcut[1]}"
      fcond.put(f"bkg_df", cond_df)

  def read_data(self, condensed_file):
    """
    Function used to read the bkg txt file
    :param condensed_file: path of the condensed file containing the saved data (either full or condensed one)
    """
    with pd.HDFStore(condensed_file, mode="r") as fcond:
      return fcond["bkg_df"]
