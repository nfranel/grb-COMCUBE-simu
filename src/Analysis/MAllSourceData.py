# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import subprocess
import multiprocessing as mp
from itertools import repeat
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import numpy as np
import glob
from pympler import asizeof
import tracemalloc
from time import time
import os

# Developped modules imports
from src.General.funcmod import printcom, printv, endtask, read_grbpar, horizon_angle, save_grb_data, eff_area_func, make_error_histogram, compile_finder, calc_trigger
from src.Catalogs.catalog import Catalog, SampleCatalog
from src.Analysis.MBkgContainer import BkgContainer
from src.Analysis.MmuSeffContainer import MuSeffContainer
from src.Analysis.MAllSimData import AllSimData
from src.Analysis.MLogData import LogData

# Ploting adjustments
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')

# plt.rcParams.update({'font.size': 20})


class AllSourceData:
  """
  Class containing all the data for a full set of trafiles
  """
  def __init__(self, grb_param, bkg_param, mu_s_eff_param, erg_cut=(100, 460), armcut=180, polarigram_bins="fixed", parallel=False, memory_check=False):
    """
    :param grb_param: str, the path to the parameter file (.par) used for the simulation
    :param bkg_param: str, the path to the parameter file (.par) used for the background simulations
    :param mu_s_eff_param: str, the path to the parameter file (.par) used for the mu100 simulation
    :param erg_cut: tuple of len 2, lower and uppuer bounds of the energy window considered
    :param armcut: int, maximum value accepted for ARM
    :param polarigram_bins: str, the way the polarigram bins are created
    :param parallel: False, int ou "all", set the parallel analysis
      False no parallelization
      int   number of cores over which the parallelization is made
      "all" all cores are used
    """
    printcom([f"Analyze of the simulation with : "
              f"   parfile : {grb_param}",
              f"   bkgparfile : {bkg_param}",
              f"   muparfile : {mu_s_eff_param}",
              f"   ergcut : {erg_cut[0]}-{erg_cut[1]}",
              f"   armcut : {armcut}"])

    printcom("Step 1 - Creating general attributes, extracting grb parfile information and compiling position finder")
    # General parameters
    self.grb_param = grb_param
    self.bkg_param = bkg_param
    self.muSeff_param = mu_s_eff_param
    self.erg_cut = erg_cut
    self.armcut = armcut

    # Parameters extracted from parfile
    self.geometry, self.revan_file, self.mimrec_file, self.simmode, self.spectra_path, self.cat_file, self.source_file, self.sim_prefix, self.sttype, self.n_sim, self.sim_duration, self.position_allowed_sim, self.sat_info = read_grbpar(self.grb_param)
    self.n_sat = len(self.sat_info)

    self.result_prefix = self.grb_param.split("/polGBM.par")[0].split("/")[-1]

    # Different kinds of bins can be made :
    if polarigram_bins in ["fixed", "limited", "optimized"]:
      self.polarigram_bins = polarigram_bins
    else:
      print("Warning : wrong option for the polarigram bins, it should be fixed (default), limited or optimized. Hence the option has been set to default value.")
      self.polarigram_bins = "fixed"
    # Setup of some options
    self.rest_cat_file = "../Data/CatData/rest_frame_properties.txt"
    self.save_time = True
    self.init_correction = False
    self.snr_min = 5
    self.options = [self.erg_cut, self.armcut, self.geometry, self.init_correction, self.polarigram_bins]
    self.dysfunctional_sats = True
    self.number_of_down_per_const = [0]

    # Memory check for the class
    if memory_check:
      print("==================================== Memory check ====================================")
      print(f"  self.grb_param memory used : {asizeof.asizeof(self.grb_param)} bytes")
      print(f"  self.bkg_param memory used : {asizeof.asizeof(self.bkg_param)} bytes")
      print(f"  self.muSeff_param memory used : {asizeof.asizeof(self.muSeff_param)} bytes")
      print(f"  self.erg_cut memory used : {asizeof.asizeof(self.erg_cut)} bytes")
      print(f"  self.armcut memory used : {asizeof.asizeof(self.armcut)} bytes")
      print(f"  self.geometry memory used : {asizeof.asizeof(self.geometry)} bytes")
      print(f"  self.revan_file memory used : {asizeof.asizeof(self.revan_file)} bytes")
      print(f"  self.mimrec_file memory used : {asizeof.asizeof(self.mimrec_file)} bytes")
      print(f"  self.simmode memory used : {asizeof.asizeof(self.simmode)} bytes")
      print(f"  self.spectra_path memory used : {asizeof.asizeof(self.spectra_path)} bytes")
      print(f"  self.cat_file memory used : {asizeof.asizeof(self.cat_file)} bytes")
      print(f"  self.source_file memory used : {asizeof.asizeof(self.source_file)} bytes")
      print(f"  self.sim_prefix memory used : {asizeof.asizeof(self.sim_prefix)} bytes")
      print(f"  self.sttype memory used : {asizeof.asizeof(self.sttype)} bytes")
      print(f"  self.n_sim memory used : {asizeof.asizeof(self.n_sim)} bytes")
      print(f"  self.sim_duration memory used : {asizeof.asizeof(self.sim_duration)} bytes")
      print(f"  self.position_allowed_sim memory used : {asizeof.asizeof(self.position_allowed_sim)} bytes")
      print(f"  self.sat_info memory used : {asizeof.asizeof(self.sat_info)} bytes")
      print(f"  self.n_sat memory used : {asizeof.asizeof(self.n_sat)} bytes")
      print(f"  self.result_prefix memory used : {asizeof.asizeof(self.result_prefix)} bytes")
      print(f"  self.polarigram_bins memory used : {asizeof.asizeof(self.polarigram_bins)} bytes")
      print(f"  self.rest_cat_file memory used : {asizeof.asizeof(self.rest_cat_file)} bytes")
      print(f"  self.save_time memory used : {asizeof.asizeof(self.save_time)} bytes")
      print(f"  self.init_correction memory used : {asizeof.asizeof(self.init_correction)} bytes")
      print(f"  self.options memory used : {asizeof.asizeof(self.options)} bytes")
      print(f"  self.dysfunctional_sats memory used : {asizeof.asizeof(self.dysfunctional_sats)} bytes")
      print(f"  self.number_of_down_per_const memory used : {asizeof.asizeof(self.number_of_down_per_const)} bytes")
      tracemalloc.start()  # Start memory monitoring
      # get memory statistics
      current, peak = tracemalloc.get_traced_memory()
      print("\nStarting memory use statistics")
      print(f"Current memory use : {current / 1024:.2f} Ko")
      print(f"Peak use : {peak / 1024:.2f} Ko")
      print("==================================== Memory check ====================================")

    # Compiling the position finder
    compile_finder()
    print("Compiling of the position finder finished")
    endtask("Step 1")

    # Setting the background files
    printcom("Step 2 - Extracting background data")
    init_time = time()
    self.bkgdata = BkgContainer(self.bkg_param, self.erg_cut)
    endtask("Step 2", timevar=init_time)

    # Setting the background files
    printcom("Step 3 - Extracting mu100 and Seff data")
    init_time = time()
    self.muSeffdata = MuSeffContainer(self.muSeff_param, self.erg_cut, self.armcut)
    endtask("Step 3", timevar=init_time)

    # Memory check for the class
    if memory_check:
      print("==================================== Memory check ====================================")
      print(f"  self.bkgdata memory used : {asizeof.asizeof(self.bkgdata)} bytes")
      print(f"  self.muSeffdata memory used : {asizeof.asizeof(self.muSeffdata)} bytes")
      # get memory statistics
      current, peak = tracemalloc.get_traced_memory()
      print("\nAfter bkg and mu100 data extraction")
      print(f"Current memory use : {current / 1024:.2f} Ko")
      print(f"Peak use : {peak / 1024:.2f} Ko")
      print("==================================== Memory check ====================================")

    # Setting the catalog and the attributes associated
    printcom("Step 4 - Loading catalog data and duty cycle information")
    init_time = time()
    if self.cat_file == "None":
      cat_data = self.extract_sources(self.sim_prefix)
      self.namelist = cat_data[0]
      self.n_source = len(self.namelist)
    else:
      if self.simmode == "GBM":
        cat_data = Catalog(self.cat_file, self.sttype, self.rest_cat_file)
      elif self.simmode == "sampled":
        cat_data = SampleCatalog(self.cat_file, self.sttype)
      else:
        raise ValueError("Wrong simulation mode in .par file")
      self.namelist = cat_data.df.name.values
      self.n_source = len(self.namelist)

      # Setting some informations used for obtaining the GRB count rates
      self.com_duty = 1  # self.n_sim_simulated / (self.n_sim_simulated + self.n_sim_in_radbelt)
      self.com_fov = 1
      if self.simmode == "GBM":
        self.cat_duration = 10
        self.gbm_duty = 0.85
        self.gbm_fov = (1 - np.cos(np.deg2rad(horizon_angle(565)))) / 2
      elif self.simmode == "sampled":
        self.cat_duration = float(self.cat_file.split("_")[-1].split("years")[0])
        self.gbm_duty = 1
        self.gbm_fov = 1
      else:
        raise ValueError("Wrong simulation mode in .par file")
      # self.com_duty = 1
      self.weights = 1 / self.n_sim / self.cat_duration * self.com_duty / self.gbm_duty * self.com_fov / self.gbm_fov
    endtask("Step 4", timevar=init_time)

    # Memory check for the class
    if memory_check:
      print("==================================== Memory check ====================================")
      print(f"  cat_data memory used : {asizeof.asizeof(cat_data)} bytes")
      print(f"  self.namelist memory used : {asizeof.asizeof(self.namelist)} bytes")
      print(f"  self.n_source memory used : {asizeof.asizeof(self.n_source)} bytes")
      print(f"  self.com_duty memory used : {asizeof.asizeof(self.com_duty)} bytes")
      print(f"  self.com_fov memory used : {asizeof.asizeof(self.com_fov)} bytes")
      print(f"  self.cat_duration memory used : {asizeof.asizeof(self.cat_duration)} bytes")
      print(f"  self.gbm_duty memory used : {asizeof.asizeof(self.gbm_duty)} bytes")
      print(f"  self.gbm_fov memory used : {asizeof.asizeof(self.gbm_fov)} bytes")
      print(f"  self.weights memory used : {asizeof.asizeof(self.weights)} bytes")
      # get memory statistics
      current, peak = tracemalloc.get_traced_memory()
      print("\nAfter loading cat")
      print(f"Current memory use : {current / 1024:.2f} Ko")
      print(f"Peak use : {peak / 1024:.2f} Ko")
      print("==================================== Memory check ====================================")

    # Log information
    printcom("Step 5 - Loading log data and simulation statistics")
    init_time = time()
    self.n_sim_simulated, self.n_sim_below_horizon, self.n_sim_in_radbelt, self.n_sin_faint, grb_names, grb_det_ites, sim_det_ites, sat_det_ites, suffix_ite = LogData(self.sim_prefix).detection_statistics(cat_data, False)
    endtask("Step 5", timevar=init_time)

    # Memory check for the class
    if memory_check:
      print("==================================== Memory check ====================================")
      print(f"  self.n_sim_simulated memory used : {asizeof.asizeof(self.n_sim_simulated)} bytes")
      print(f"  self.n_sim_below_horizon memory used : {asizeof.asizeof(self.n_sim_below_horizon)} bytes")
      print(f"  self.n_sim_in_radbelt memory used : {asizeof.asizeof(self.n_sim_in_radbelt)} bytes")
      print(f"  self.n_sin_faint memory used : {asizeof.asizeof(self.n_sin_faint)} bytes")
      print(f"  grb_names memory used : {asizeof.asizeof(grb_names)} bytes")
      print(f"  grb_det_ites memory used : {asizeof.asizeof(grb_det_ites)} bytes")
      print(f"  sim_det_ites memory used : {asizeof.asizeof(sim_det_ites)} bytes")
      print(f"  sat_det_ites memory used : {asizeof.asizeof(sat_det_ites)} bytes")
      print(f"  suffix_ite memory used : {asizeof.asizeof(suffix_ite)} bytes")
      # get memory statistics
      current, peak = tracemalloc.get_traced_memory()
      print("\nAfter loading logfile")
      print(f"Current memory use : {current / 1024:.2f} Ko")
      print(f"Peak use : {peak / 1024:.2f} Ko")
      print("==================================== Memory check ====================================")

    # Extracting the information from the simulation files
    printcom("Step 6 - preparing filenames for simulation files and extracted simulation files")
    init_time = time()

    if not os.path.exists(f"{self.sim_prefix.split('/sim/')[0]}/extracted"):
      os.mkdir(f"{self.sim_prefix.split('/sim/')[0]}/extracted")
    tobe_extracted, extracted_name, presence_list, filtered_ites = self.filenames_creation(grb_names, grb_det_ites, sim_det_ites, sat_det_ites, suffix_ite)
    num_files = int(subprocess.getoutput(f"ls {self.sim_prefix.split('/sim/')[0]}/sim | wc").strip().split("  ")[0])
    if num_files > self.n_sim_simulated:
      print("ERROR : The number of file in the log is smaller than the number of files")
    elif num_files < self.n_sim_simulated:
      print("ERROR : The number of file in the log is greater than the number of files")
      print("The missing files are : ")
      for total_path in tobe_extracted:
        if not os.path.exists(total_path):
          print(f"  {total_path.split('/sim/')[-1]}")
      raise FileNotFoundError
    endtask("Step 6", timevar=init_time)

    if memory_check:
      print("==================================== Memory check ====================================")
      print(f"  tobe_extracted memory used : {asizeof.asizeof(tobe_extracted)} bytes")
      print(f"  extracted_name memory used : {asizeof.asizeof(extracted_name)} bytes")
      print(f"  presence_list memory used : {asizeof.asizeof(presence_list)} bytes")
      print(f"  num_files memory used : {asizeof.asizeof(num_files)} bytes")
      # get memory statistics
      current, peak = tracemalloc.get_traced_memory()
      print("\nAfter preparing filenames")
      print(f"Current memory use : {current / 1024:.2f} Ko")
      print(f"Peak use : {peak / 1024:.2f} Ko")
      print("==================================== Memory check ====================================")

    printcom("Step 7 - Extracting the information from the simulation files")
    init_time = time()

    # Extracting the information from the simulation files
    if parallel == 'all':
      print("Parallel extraction of the data with all threads")
      with mp.Pool() as pool:
        pool.starmap(save_grb_data, zip(tobe_extracted, extracted_name, repeat(self.sat_info), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.geometry)))
    elif type(parallel) is int:
      print(f"Parallel extraction of the data with {parallel} threads")
      with mp.Pool(parallel) as pool:
        pool.starmap(save_grb_data, zip(tobe_extracted, extracted_name, repeat(self.sat_info), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.geometry)))
    else:
      [save_grb_data(tobe_extracted[ext_ite], extracted_name[ext_ite], self.sat_info, self.bkgdata, self.muSeffdata, self.geometry) for ext_ite in range(len(tobe_extracted))]
    endtask("Step 7", timevar=init_time)

    printcom("Step 8 - Loading GRB extracted data")
    init_time = time()
    # Reading the information from the extracted simulation files
    if parallel == 'all':
      print("Parallel extraction of the data with all threads")
      with mp.Pool() as pool:
        self.alldata = pool.starmap(AllSimData, zip(presence_list, filtered_ites, repeat(cat_data), repeat(self.sat_info), repeat(self.sim_duration), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.options)))
    elif type(parallel) is int:
      print(f"Parallel extraction of the data with {parallel} threads")
      with mp.Pool(parallel) as pool:
        self.alldata = pool.starmap(AllSimData, zip(presence_list, filtered_ites, repeat(cat_data), repeat(self.sat_info), repeat(self.sim_duration), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.options)))
    else:
      self.alldata = [AllSimData(presence_list[source_ite], filtered_ites[source_ite], cat_data, self.sat_info, self.sim_duration, self.bkgdata, self.muSeffdata, self.options) for source_ite in range(self.n_source)]
    endtask("Step 8", timevar=init_time)

    if memory_check:
      print("==================================== Memory check ====================================")
      print(f"  self.alldata memory used : {asizeof.asizeof(self.alldata)} bytes")
      # get memory statistics
      current, peak = tracemalloc.get_traced_memory()
      print("\nAfter extracting data")
      print(f"Current memory use : {current / 1024:.2f} Ko")
      print(f"Peak use : {peak / 1024:.2f} Ko")
      tracemalloc.stop()
      print("==================================== Memory check ====================================")

  def filenames_creation(self, grb_names, grb_det_ites, sim_det_ites, sat_det_ites, suffix_ite):
    tobe_ext = []
    ext_name = []
    pres_list = np.empty((self.n_source, self.n_sim, self.n_sat), dtype=object)
    # pres_list = np.empty((len(grb_names), self.n_sim, self.n_sat), dtype=object)
    for ite, grbname in enumerate(grb_names):
      temp_simfile = f"{self.sim_prefix}_{grbname}_sat{sat_det_ites[ite]}_{sim_det_ites[ite]:04d}_{suffix_ite[ite]}.inc1.id1.extracted.tra"
      tobe_ext.append(temp_simfile)
      temp_name = f"{self.sim_prefix.split('/sim/')[0]}/extracted/{self.sim_prefix.split('/sim/')[1]}_extracted{grbname}_sat{sat_det_ites[ite]}_{sim_det_ites[ite]:04d}.h5"
      ext_name.append(temp_name)
      pres_list[grb_det_ites[ite]][sim_det_ites[ite]][sat_det_ites[ite]] = temp_name
    pres_list = pres_list.tolist()
    final_pres_list = []
    final_ites = []
    for ite in range(len(pres_list)):
      if not np.all([val is None for val in np.array(pres_list[ite]).flatten()]):
        final_pres_list.append(pres_list[ite])
        final_ites.append(ite)
    return tobe_ext, ext_name, final_pres_list, final_ites

  # TODO finish the comments and rework the methods !
  def extract_sources(self, prefix, duration=None):
    """
    Function used when the simulations are not comming from GBM GRB data (ex Crab nebula, etc)
    :param prefix: Prefix used for the simulation file
    :param duration: Specific duration given to the source (option used for tests so far)
    :returns: a list containing a list of the source names and a list of their duration
              (Here the duration is always the same, may be changed)
    """
    if duration is None:
      if self.sim_duration.isdigit():
        duration = float(self.sim_duration)
      elif self.sim_duration == "t90" or self.sim_duration == "lc":
        duration = None
        print("Warning : impossible to load the t90 as no catalog is given.")
      else:
        duration = None
        print("Warning : unusual sim duration, please check the parameter file.")

    # flist = subprocess.getoutput(f"ls {prefix}_*").split("\n")
    flist = glob.glob(f"{prefix}_*")
    source_names = []
    if len(flist) >= 1 and not flist[0].startswith("ls: cannot access"):
      temp_sourcelist = []
      for file in flist:
        temp_sourcelist.append(file.split("_")[1])
      source_names = list(set(temp_sourcelist))
    return [source_names, [duration] * len(source_names)]

  def azi_angle_corr(self):
    """
    Method to apply the angle correction to all polarigrams (So that they are in a same referential)
    """
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            for sat in sim:
              if sat is not None:
                sat.corr()

  def azi_angle_anticorr(self):
    """
    Method to remove the angle correction to all polarigrams
    """
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            for sat in sim:
              if sat is not None:
                sat.anticorr()

  def analyze(self, sats_analysis=False):
    """
    Proceed to the analysis of polarigrams for all satellites and constellation (unless specified) for all data
    and calculates some probabilities
    :param sats_analysis: True if the analysis is done both on the sat data and on the constellation data
    """
    printcom("Analyze of the data - Analyzing the data after extraction and creation of the constellations")
    printcom("Be aware that the snrs and mdp calculated for the constellation may not use data from all satellites in sight because of beneficial_compton and beneficial_single options")
    init_time = time()
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim_ite, sim in enumerate(source):
          if sim is not None:
            sim.analyze(source.source_duration, source.source_fluence, sats_analysis)
        # source.set_probabilities(n_sat=self.n_sat, snr_min=self.snr_min, n_image_min=50)  # todo change it
    endtask("Analyze of the data", timevar=init_time)

  def set_beneficial(self, threshold_mdp):
    """
    Sets const_beneficial_compton and const_beneficial_compton to True is the value for a satellite is worth considering
    """
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim_ite, sim in enumerate(source):
          if sim is not None:
            for sat in sim:
              if sat is not None:
                sat.set_beneficial_compton(threshold=threshold_mdp)
                sat.set_beneficial_single()

  def make_const(self, condensed_const=True, const=None):
    """
    This function is used to combine results from different satellites
    Results are then stored in the key const_data
    ! The polarigrams have to be corrected to combine the polarigrams !
    :param const: Which satellite are considered for the constellation if none, all of them are
    """
    printcom("Creation of the constellations")
    init_time = time()
    ###################################################################################################################
    # Setting some satellites off
    ###################################################################################################################
    off_sats = []  # TODO put a verification to see if there is at least 1 value in the list number_of_down_per_const
    for num_down in self.number_of_down_per_const:
      if num_down == 0:
        off_sats.append(None)
      else:
        temp_offsat = []
        while len(temp_offsat) != num_down:
          rand_sat = np.random.randint(self.n_sat)
          if rand_sat not in temp_offsat:
            temp_offsat.append(rand_sat)
        off_sats.append(temp_offsat)
    ###################################################################################################################
    # Making of the constellations  -  with corrected polarigrams (if not corrected they can't be added)
    ###################################################################################################################
    if not self.init_correction:
      self.azi_angle_corr()
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if condensed_const:
              sim.make_condensed_const(self.number_of_down_per_const, off_sats, const=const, dysfunction_enabled=self.dysfunctional_sats)
            else:
              sim.make_const(self.number_of_down_per_const, off_sats, const=const, dysfunction_enabled=self.dysfunctional_sats)
    if not self.init_correction:
      self.azi_angle_anticorr()
    endtask("Creation of the constellations", timevar=init_time)

  def source_search(self, source_name, verbose=True):
    """
    Search among the sources simulated if one has the correct name
    :param source_name: The name of the source searched
    :param verbose: verbosity to print the results or just return the index of the source in the catalog
    :returns: the position of the source(s) in the list and displays other information unless specified if it's there
    """
    printv("================================================", verbose)
    printv("==            Searching the source            ==", verbose)
    source_position = np.where(np.array(self.namelist) == source_name)[0]
    if len(source_position) == 0:
      printv(f"No source corresponding to {source_name}, returning None", verbose)
      return None
    elif len(source_position) > 1:
      printv(f"Several items have been found that matches the source name {source_name}, returning a list of the indices", verbose)
      return source_position
    elif len(source_position) == 1:
      printv(f"The source {source_name} has been found at position {source_position[0]}, returning this position as an integer", verbose)
      printv("==  Additionnal information about the source  ==", verbose)
      printv(f" - Source duration : {self.alldata[source_position[0]].source_duration} s", verbose)
      printv(f" - Source flux at peak : {self.alldata[source_position[0]].best_fit_p_flux} photons/cm²/s", verbose)
      printv(f" - Source fluence between {self.erg_cut[0]} keV and {self.erg_cut[1]} keV : {self.alldata[source_position[0]].source_fluence} photons/cm²", verbose)
      printv(f" - Number of simulation in the constellation's field of view : {len(self.alldata[source_position[0]])}", verbose)
      return source_position[0]

  def source_information(self, source_id, verbose=True):
    """
    Give information about a given source. If source_id is a name then proceeds to a source search beforehand
    :param source_id: The index or name of the source we want to know about
    :param verbose: verbosity to print the results if false nothing is shown
    """
    if type(source_id) is int:
      source_ite = source_id
      source = self.alldata[source_ite]
    elif type(source_id) is str:
      printv("================================================", verbose)
      printv("==            Searching the source            ==", verbose)
      source_position = np.where(np.array(self.namelist) == source_id)[0]
      if len(source_position) == 0:
        printv(f"No source corresponding to {source_id}, returning None", verbose)
        return None
      elif len(source_position) > 1:
        printv(
          f"Several items have been found that matches the source name {source_id}, returning a list of the indices",
          verbose)
        return source_position
      elif len(source_position) == 1:
        source_ite = source_position[0]
        source = self.alldata[source_ite]
      else:
        return None
    else:
      printv(f"The source id doesn't match any known structure, use the position of the source in the list or its name.", verbose)
      return None
    printv(f"The source {self.namelist[source_ite]} is at position {source_ite} in the data list", verbose)
    printv("==  General information about the source simulated  ==", verbose)
    printv(f" - Source duration : {source.source_duration} s", verbose)
    printv(f" - Source spectrum model : {source.best_fit_model}", verbose)
    printv(f" - Source flux at peak (best fit) : {source.best_fit_p_flux} photons/cm²/s", verbose)
    printv(f" - Source mean flux (best fit) : {source.best_fit_mean_flux} photons/cm²/s", verbose)
    printv(f" - Source fluence between {self.erg_cut[0]} keV and {self.erg_cut[1]} keV : {source.source_fluence} photons/cm²", verbose)
    printv(f" - Number of simulation in the constellation's field of view : {len(source)}", verbose)
    printv("==  Precise information about the simulation and satellites  ==", verbose)
    for sim_ite, sim in enumerate(source):
      if sim is not None:
        printv(f" - For simulation {sim_ite} in the constellation's field of view :", verbose)
        for sat_ite, sat in enumerate(sim):
          if sat is not None:
            printv(f" - For satellite {sat_ite} :", verbose)
            printv(f"   - Position : DEC - {sat.dec_sat_frame}     RA - {sat.ra_sat_frame}", verbose)
          else:
            printv(f" - Satellite {sat_ite} do not see the source.", verbose)
      else:
        printv(f" - For simulation {sim_ite} not in the field of view.", verbose)
    return None

  def study_mdp_threshold(self, mdp_thresh_list, savefile=None):
    """
    Give the mdp results for several MDP threshold to study its influence on the performances
    """
    # Search for a mdp limit :
    if savefile is None:
      for threshold_mdp in mdp_thresh_list:
        self.set_beneficial(threshold_mdp)
        self.make_const()
        self.analyze()
        print(f" ========               MDP THRESHOLD USED : {threshold_mdp}   ========")
        number_detected = 0
        mdp_list = []
        for source in self.alldata:
          if source is not None:
            for sim in source:
              if sim is not None:
                if sim.const_data is not None:
                  number_detected += 1
                  if sim.const_data[0].mdp is not None:
                    if sim.const_data[0].mdp <= 1:
                      mdp_list.append(sim.const_data[0].mdp * 100)
        mdp_list = np.array(mdp_list)
        print("=                        MDP detection rates                        =")
        print(f"   MDP<=100% : {np.sum(np.where(mdp_list <= 100, 1, 0)) * self.weights}")
        print(f"   MDP<=90%  : {np.sum(np.where(mdp_list <= 90, 1, 0)) * self.weights}")
        print(f"   MDP<=80%  : {np.sum(np.where(mdp_list <= 80, 1, 0)) * self.weights}")
        print(f"   MDP<=70%  : {np.sum(np.where(mdp_list <= 70, 1, 0)) * self.weights}")
        print(f"   MDP<=60%  : {np.sum(np.where(mdp_list <= 60, 1, 0)) * self.weights}")
        print(f"   MDP<=50%  : {np.sum(np.where(mdp_list <= 50, 1, 0)) * self.weights}")
        print(f"   MDP<=40%  : {np.sum(np.where(mdp_list <= 40, 1, 0)) * self.weights}")
        print(f"   MDP<=30%  : {np.sum(np.where(mdp_list <= 30, 1, 0)) * self.weights}")
        print(f"   MDP<=20%  : {np.sum(np.where(mdp_list <= 20, 1, 0)) * self.weights}")
        print(f"   MDP<=10%  : {np.sum(np.where(mdp_list <= 10, 1, 0)) * self.weights}")
    elif type(savefile) is str:
      with open(savefile, "w") as f:
        f.write("Result file for the MDP threshold study\n")
        f.write("Threshold | MDP100 | MDP90 | MDP80 | MDP70 | MDP60 | MDP50 | MDP40 | MDP30 | MDP20 | MDP10 | Number detected | Number mdp <= 100\n")
        for threshold_mdp in mdp_thresh_list:
          self.set_beneficial(threshold_mdp)
          self.make_const()
          self.analyze()
          print(f" ========               MDP THRESHOLD USED : {threshold_mdp}   ========")
          number_detected = 0
          mdp_list = []
          for source in self.alldata:
            if source is not None:
              for sim in source:
                if sim is not None:
                  if sim.const_data is not None:
                    number_detected += 1
                    if sim.const_data[0].mdp is not None:
                      if sim.const_data[0].mdp <= 1:
                        mdp_list.append(sim.const_data[0].mdp * 100)
          mdp_list = np.array(mdp_list)
          mdp100 = np.sum(np.where(mdp_list <= 100, 1, 0)) * self.weights
          mdp90 = np.sum(np.where(mdp_list <= 90, 1, 0)) * self.weights
          mdp80 = np.sum(np.where(mdp_list <= 80, 1, 0)) * self.weights
          mdp70 = np.sum(np.where(mdp_list <= 70, 1, 0)) * self.weights
          mdp60 = np.sum(np.where(mdp_list <= 60, 1, 0)) * self.weights
          mdp50 = np.sum(np.where(mdp_list <= 50, 1, 0)) * self.weights
          mdp40 = np.sum(np.where(mdp_list <= 40, 1, 0)) * self.weights
          mdp30 = np.sum(np.where(mdp_list <= 30, 1, 0)) * self.weights
          mdp20 = np.sum(np.where(mdp_list <= 20, 1, 0)) * self.weights
          mdp10 = np.sum(np.where(mdp_list <= 10, 1, 0)) * self.weights

          f.write(f"{threshold_mdp} | {mdp100} | {mdp90} | {mdp80} | {mdp70} | {mdp60} | {mdp50} | {mdp40} | {mdp30} | {mdp20} | {mdp10} | {number_detected} | {len(mdp_list)}\n")

          print("=                        MDP detection rates                        =")
          print(f"   MDP<=100% : {mdp100}")
          print(f"   MDP<=90%  : {mdp90}")
          print(f"   MDP<=80%  : {mdp80}")
          print(f"   MDP<=70%  : {mdp70}")
          print(f"   MDP<=60%  : {mdp60}")
          print(f"   MDP<=50%  : {mdp50}")
          print(f"   MDP<=40%  : {mdp40}")
          print(f"   MDP<=30%  : {mdp30}")
          print(f"   MDP<=20%  : {mdp20}")
          print(f"   MDP<=10%  : {mdp10}")

    else:
      print("Type error for savefile, must be str or None")

  def count_triggers(self, const_index=0, parallel=10, graphs=False, lc_aligned=False):
    """
    Function to count and print the number of triggers using different criterions
    """
    print("================================================================================================")
    print(f"== Triggers according to GBM method with   {self.number_of_down_per_const[const_index]}   down satellite")
    print("================================================================================================")
    if type(parallel) == int:
      print(f"Parallel extraction of the data with {parallel} threads")
      with mp.Pool(parallel) as pool:
        ret = pool.starmap(calc_trigger, zip(self.alldata, range(len(self.alldata)), repeat(const_index), repeat(lc_aligned)))
    else:
      raise TypeError("Parameter parallel must be an int")
    [trigg_1s, trigg_2s, trigg_3s, trigg_4s, no_trig_name, no_trig_duration, no_trig_dec, no_trig_e_fluence] = np.array(ret).transpose()
    total_in_view = 0
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            total_in_view += 1

    trigg_1s = trigg_1s[~np.isnan(trigg_1s.astype(float))]
    trigg_2s = trigg_2s[~np.isnan(trigg_2s.astype(float))]
    trigg_3s = trigg_3s[~np.isnan(trigg_3s.astype(float))]
    trigg_4s = trigg_4s[~np.isnan(trigg_4s.astype(float))]

    # const_trigger_counter_4s = np.count_nonzero(~np.isnan(trigg_4s.astype(float)))
    # const_trigger_counter_3s = np.count_nonzero(~np.isnan(trigg_3s.astype(float)))
    # const_trigger_counter_2s = np.count_nonzero(~np.isnan(trigg_2s.astype(float)))
    # const_trigger_counter_1s = np.count_nonzero(~np.isnan(trigg_1s.astype(float)))

    const_trigger_counter_4s = np.count_nonzero(trigg_4s)
    const_trigger_counter_3s = np.count_nonzero(trigg_3s)
    const_trigger_counter_2s = np.count_nonzero(trigg_2s)
    const_trigger_counter_1s = np.count_nonzero(trigg_1s)

    print(f"   Trigger for at least 4 satellites :        {const_trigger_counter_4s:.2f} triggers")
    print(f"   Trigger for at least 3 satellites :        {const_trigger_counter_3s:.2f} triggers")
    print(f"   Trigger for at least 2 satellites :        {const_trigger_counter_2s:.2f} triggers")
    print(f"   Trigger for 1 satellite :        {const_trigger_counter_1s:.2f} triggers")
    print("=============================================")
    print(f" Over the {total_in_view} GRBs simulated in the constellation field of view")
    if graphs:
      no_trig_duration = np.array(no_trig_duration, dtype=float)
      no_trig_dec = np.array(no_trig_dec, dtype=float)
      no_trig_e_fluence = np.array(no_trig_e_fluence, dtype=float)

      t1 = f"Not triggered GRB distribution with {self.number_of_down_per_const[const_index]} satellite down\n{len(no_trig_duration)} not triggered over {total_in_view} GRB simulated"
      fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 6))
      fig.suptitle(t1)
      ax1.hist(no_trig_duration, bins=20, histtype="step")
      ax1.set(xlabel="GRB duration (s)", ylabel="Number of not triggered", xscale="linear", yscale="linear")

      ax2.hist(no_trig_dec, bins=20, histtype="step")
      ax2.set(xlabel="GRB dec in world frame (°)", ylabel="Number of not triggered", xscale="linear", yscale="linear")

      ax3.hist(no_trig_e_fluence, bins=np.logspace(int(np.log10(np.min(no_trig_e_fluence))), int(np.log10(np.max(no_trig_e_fluence)) + 1), 20), histtype="step")
      ax3.set(xlabel="GRB energy fluence (erg/cm²)", ylabel="Number of not triggered", xscale="log", yscale="linear")
      plt.show()
    return trigg_1s, trigg_2s, trigg_3s, trigg_4s

  def fov_const(self, num_val=500, show=True, save=False):
    """
    Plots a map of the sensibility over the sky for number of sat in sight, single events and compton events
    :param num_val: number of value to
    """
    plt.rcParams.update({'font.size': 15})
    plt.tight_layout()
    xlab = "Right ascention (°)"
    ylab = "Declination (°)"
    title1 = "Constellation sky coverage map"
    title2 = "Constellation sky sensitivity map for Compton events"
    title3 = "Constellation sky sensitivity map for single events"
    bar1 = "Number of satellites covering the area"
    bar2 = "Effective area for Compton events (cm²)"
    bar3 = "Effective area for single events (cm²)"
    chosen_proj = "mollweide"

    phi_world = np.linspace(0, 360, num_val, endpoint=False)
    # theta will be converted in sat coord with grb_decra_worldf2satf, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, 180, num_val)
    detection = np.zeros((self.n_sat, num_val, num_val))
    detection_compton = np.zeros((self.n_sat, num_val, num_val))
    detection_single = np.zeros((self.n_sat, num_val, num_val))

    nite = num_val ** 2 * len(self.sat_info)
    ncount = 0
    for ite, info_sat in enumerate(self.sat_info):
      for ite_theta, theta in enumerate(theta_world):
        for ite_phi, phi in enumerate(phi_world):
          ncount += 1
          detection_compton[ite][ite_theta][ite_phi], detection_single[ite][ite_theta][ite_phi], detection[ite][ite_theta][ite_phi] = eff_area_func(theta, phi, info_sat, self.muSeffdata)
          print(f"Calculation : {int(ncount / nite * 100)}%", end="\r")
    print("Calculation over")

    detec_sum = np.sum(detection, axis=0)
    detec_sum_compton = np.sum(detection_compton, axis=0)
    detec_sum_single = np.sum(detection_single, axis=0)

    phi_plot, theta_plot = np.meshgrid(np.deg2rad(phi_world) - np.pi, np.pi / 2 - np.deg2rad(theta_world))
    detec_min = int(np.min(detec_sum))
    detec_max = int(np.max(detec_sum))
    detec_min_compton = int(np.min(detec_sum_compton))
    detec_max_compton = int(np.max(detec_sum_compton))
    detec_min_single = int(np.min(detec_sum_single))
    detec_max_single = int(np.max(detec_sum_single))
    cmap_det = mpl.cm.Blues_r
    cmap_compton = mpl.cm.Greens_r
    cmap_single = mpl.cm.Oranges_r

    ##################################################################################################################
    # Map for number of satellites in sight
    ##################################################################################################################
    levels = range(detec_min, detec_max + 1, max(1, int((detec_max + 1 - detec_min) / 15)))

    fig1, ax1 = plt.subplots(subplot_kw={'projection': chosen_proj}, figsize=(15, 8))
    # ax1.set_global()
    # ax1.coastlines()
    h1 = ax1.pcolormesh(phi_plot, theta_plot, detec_sum, cmap=cmap_det)
    # ax1.axis('scaled')
    ax1.set(xlabel=xlab, ylabel=ylab, title=title1)
    cbar = fig1.colorbar(h1, ticks=levels)
    cbar.set_label(bar1, rotation=270, labelpad=20)
    if save:
      fig1.savefig(f"{self.result_prefix}_in_sight_erg{self.erg_cut[0]}-{self.erg_cut[1]}")
    if show:
      plt.show()

    ##################################################################################################################
    # Map of constellation's compton effective area
    ##################################################################################################################
    levels_compton = range(detec_min_compton, detec_max_compton + 1, max(1, int((detec_max_compton + 1 - detec_min_compton) / 15)))

    # fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    fig2, ax2 = plt.subplots(subplot_kw={'projection': chosen_proj}, figsize=(15, 8))
    # ax2.set_global()
    # ax2.coastlines()
    h3 = ax2.pcolormesh(phi_plot, theta_plot, detec_sum_compton, cmap=cmap_compton)
    # ax2.axis('scaled')
    ax2.set(xlabel=xlab, ylabel=ylab, title=title2)
    cbar = fig2.colorbar(h3, ticks=levels_compton)
    cbar.set_label(bar2, rotation=270, labelpad=20)
    if save:
      fig2.savefig(f"{self.result_prefix}_compton_seff_erg{self.erg_cut[0]}-{self.erg_cut[1]}")
    if show:
      plt.show()

    ##################################################################################################################
    # Map of constellation's single effective area
    ##################################################################################################################
    levels_single = range(detec_min_single, detec_max_single + 1, max(1, int((detec_max_single + 1 - detec_min_single) / 15)))

    fig3, ax3 = plt.subplots(subplot_kw={'projection': chosen_proj}, figsize=(15, 8))
    # ax3.set_global()
    # ax3.coastlines()
    h5 = ax3.pcolormesh(phi_plot, theta_plot, detec_sum_single, cmap=cmap_single)
    # ax3.axis('scaled')
    ax3.set(xlabel=xlab, ylabel=ylab, title=title3)
    cbar = fig3.colorbar(h5, ticks=levels_single)
    cbar.set_label(bar3, rotation=270, labelpad=20)
    if save:
      fig3.savefig(f"{self.result_prefix}_single_seff_erg{self.erg_cut[0]}-{self.erg_cut[1]}")
    if show:
      plt.show()

    correction_values = (1 + np.sin(np.deg2rad(theta_world)) * (num_val - 1)) / num_val
    # print(f"The mean number of satellites in sight is :       {np.mean(np.mean(detec_sum, axis=1) * correction_values):.4f} satellites")
    # print(f"The mean effective area for Compton events is :  {np.mean(np.mean(detec_sum_compton, axis=1) * correction_values):.4f} cm²")
    # print(f"The mean effective area for single events is :   {np.mean(np.mean(detec_sum_single, axis=1) * correction_values):.4f} cm²")

    print(f"The mean number of satellites in sight is :       {np.average(np.mean(detec_sum, axis=1), weights=correction_values):.4f} satellites")
    print(f"The mean effective area for Compton events is :  {np.average(np.mean(detec_sum_compton, axis=1), weights=correction_values):.4f} cm²")
    print(f"The mean effective area for single events is :   {np.average(np.mean(detec_sum_single, axis=1), weights=correction_values):.4f} cm²")

    # print(f"NOT SIN CORRECTED - The mean number of satellites in sight is :       {np.mean(detec_sum):.4f} satellites")
    # print(f"NOT SIN CORRECTED - The mean effective area for Compton events is :  {np.mean(detec_sum_compton):.4f} cm²")
    # print(f"NOT SIN CORRECTED - The mean effective area for single events is :   {np.mean(detec_sum_single):.4f} cm²")

  def grb_map_plot(self, mode="no_cm"):
    """
    Display the catalog GRBs position in the sky using the corresponding function in catalog.py
    :param mode: mode to give colormap options to the plot
      mode can be "no_cm" or "t90"
    """
    cat_data = Catalog(self.cat_file, self.sttype, self.rest_cat_file)
    cat_data.grb_map_plot(mode)

  def spectral_information(self):
    """
    Displays the spectral information of the GRBs including the proportion of different best fit models and the
    corresponding parameters
    """
    cat_data = Catalog(self.cat_file, self.sttype, self.rest_cat_file)
    cat_data.spectral_information()

  def mdp_histogram(self, const_index=0, mdp_limit=1, cumul=1, n_bins=30, x_scale='linear', y_scale="log"):
    """
    Display and histogram representing the number of grb of a certain mdp per year
    :param selected_sat: int or string, which sat is selected, if "const" the constellation is selected
    :param mdp_limit: limit in mdp (mdp more than 1 is not physical so should be between 0 and 1)
    :param cumul: int, 1 for a cumulative histogram, 0 for a usual one
    :param n_bins: number of bins in the histogram
    :param x_scale: scale for x-axis
    :param y_scale: scale for y-axis
    """
    if self.cat_file.endswith("longGBM.txt"):
      grb_type = "lGRB"
    elif self.cat_file.endswith("shortGRB.txt"):
      grb_type = "sGRB"
    elif self.cat_file.endswith("allGRB.txt"):
      grb_type = "all GRB"
    else:
      grb_type = "undefined source"
    number_detected = 0
    mdp_list = []
    mdp_err_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if sim.const_data[const_index] is not None:
              number_detected += 1
              if sim.const_data[const_index].mdp is not None:
                if sim.const_data[const_index].mdp <= mdp_limit:
                  mdp_list.append(sim.const_data[const_index].mdp * 100)
                  mdp_err_list.append(sim.const_data[const_index].mdp_err * 100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    h1 = ax.hist(mdp_list, bins=n_bins, cumulative=cumul, histtype="step", weights=[self.weights] * len(mdp_list),
            label=f"Number of GRBs with MDP < {mdp_limit * 100}% : {len(mdp_list)} over {number_detected} detections", color="blue")
    mdp_errinf, mdp_errsup = make_error_histogram(np.array(mdp_list), np.array(mdp_err_list), h1[1])
    if cumul == 1:
      ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Cumulative distribution of the MDP - {grb_type}")
      new_errinf = np.copy(mdp_errinf)
      new_errsup = np.copy(mdp_errsup)
      for ite in range(1, len(mdp_errinf)):
        new_errinf[ite:] += mdp_errinf[ite - 1]
        new_errsup[ite:] += mdp_errsup[ite - 1]
      mdp_inf, mdp_sup = h1[0] + new_errinf * self.weights, h1[0] + new_errsup * self.weights
    elif cumul == 0:
      ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Distribution of the MDP - {grb_type}")
      mdp_inf, mdp_sup = h1[0] + mdp_errinf * self.weights, h1[0] + mdp_errsup * self.weights
    else:
      raise ValueError("Use a correct value for cumul, only 1 and 0 work")
    mdp_inf, mdp_sup = np.concatenate((mdp_inf, np.array([0]))), np.concatenate((mdp_sup, np.array([0])))
    ax.fill_between(h1[1], mdp_inf, mdp_sup, step="post", alpha=0.4, color="blue")
    ax.legend(loc='upper left')
    ax.grid(axis='both')
    plt.show()

  def snr_histogram(self, snr_type="compton", const_index=0, cumul=0, n_bins=30, x_scale="log", y_scale="log"):
    """
    Display and histogram representing the number of grb that have at least a certain snr per year
    :param snr_type: "compton" or "single" to consider either compton events or single events
    :param selected_sat: int or string, which sat is selected, if "const" the constellation is selected
    :param cumul: int, 1 for a cumulative histogram, 0 for a usual one, -1 for an inverse cumulative one
    :param n_bins: number of bins in the histogram
    :param x_scale: scale for x-axis
    :param y_scale: scale for y-axis
    """
    if self.cat_file.endswith("longGBM.txt"):
      grb_type = "lGRB"
    elif self.cat_file.endswith("shortGRB.txt"):
      grb_type = "sGRB"
    elif self.cat_file.endswith("allGRB.txt"):
      grb_type = "all GRB"
    else:
      grb_type = "undefined source"
    if snr_type == "compton":
      x_label = f"SNR for compton events (dimensionless)"
    elif snr_type == "single":
      x_label = f"SNR for single events (dimensionless)"
    else:
      print("Choose a correct type of snr : compton (default) or single ")
      return "snr_histogram error"
    snr_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if snr_type == "compton":
              snr_list.append(sim.const_data[const_index].snr_compton_t90)
            elif snr_type == "single":
              snr_list.append(sim.const_data[const_index].snr_single_t90)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if x_scale == "log":
      if min(snr_list) < 1:
        inf_limit = int(np.log10(min(snr_list))) - 1
      else:
        inf_limit = int(np.log10(min(snr_list)))
      if max(snr_list) > 1:
        sup_limit = int(np.log10(max(snr_list))) + 1
      else:
        sup_limit = int(np.log10(max(snr_list)))
      hist_bins = np.logspace(inf_limit, sup_limit, n_bins)
    else:
      hist_bins = n_bins
    ax.hist(snr_list, bins=hist_bins, cumulative=cumul, histtype="step", weights=[self.weights] * len(snr_list))
    if cumul == 1:
      ax.set(xlabel=x_label, ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Cumulative distribution of the SNR - {grb_type}")
    elif cumul == 0:
      ax.set(xlabel=x_label, ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Distribution of the SNR - {grb_type}")
    elif cumul == -1:
      ax.set(xlabel=x_label, ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Inverse cumulative distribution of the SNR - {grb_type}")
    ax.grid(axis='both')
    plt.show()

  def brightest_det_stats(self, n_grb, lc_plot=True, det_max_repartition_plot=False):
    lst_pflux = []
    combined_sats_det_stats_shaped = None
    # if self.simmode == "GBM":
    #   cat_data = Catalog(self.cat_file, self.sttype, self.rest_cat_file)
    # elif self.simmode == "sampled":
    #   cat_data = SampleCatalog(self.cat_file, self.sttype)
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        if source.best_fit_p_flux is not None:
          lst_pflux.append(source.best_fit_p_flux)
        else:
          lst_pflux.append(0)
    lst_pflux = np.array(lst_pflux, dtype=np.float32)
    idxs = np.argsort(lst_pflux)[::-1][:n_grb]
    for idx in idxs:
      if self.alldata[idx] is not None:
        for sim in self.alldata[idx]:
          if sim is not None:
            max_sat_idx = np.argmax(np.array([sat.compton * 2 + sat.single if sat is not None else 0 for sat in sim]))
            sats_det_stats = []
            sat_id_list = []
            for ite_sat, sat in enumerate(sim):
              if sat is not None:
                sat_id_list.append(ite_sat)
                if ite_sat == max_sat_idx:
                  sats_det_stats.append(sat.detector_statistics(self.bkgdata.bkgdf.iloc[sat.bkg_index], self.bkgdata.sim_time, self.alldata[idx].source_duration, self.alldata[idx].source_name, show=lc_plot))
                else:
                  sats_det_stats.append(sat.detector_statistics(self.bkgdata.bkgdf.iloc[sat.bkg_index], self.bkgdata.sim_time, self.alldata[idx].source_duration, self.alldata[idx].source_name, show=False))
            sats_det_stats_shaped = np.transpose(np.array(sats_det_stats), (1, 2, 0))
            if self.alldata[idx].source_name == "GRB130427324":
              print(f"Max from brightest GRB - GRB130427324 : {np.max(sats_det_stats_shaped)}")
            if combined_sats_det_stats_shaped is None:
              combined_sats_det_stats_shaped = sats_det_stats_shaped
            else:
              combined_sats_det_stats_shaped = np.concatenate((combined_sats_det_stats_shaped, sats_det_stats_shaped), axis=2)
            # ! test the shape change, the detector stat gathering and the graphs
            if det_max_repartition_plot:
              fig2, axes2 = plt.subplots(4, 5)
              fig2.suptitle(f"Detectors max count rate - {self.alldata[idx].source_name} at peak for different satellites")
              axes2[0, 0].set(ylabel="Quad1\nNumber of the satellite")
              axes2[1, 0].set(ylabel="Quad2\nNumber of the satellite")
              axes2[2, 0].set(ylabel="Quad3\nNumber of the satellite")
              axes2[3, 0].set(xlabel="Detector max count rate (hit/s)\nSideDetX", ylabel="Quad4\nNumber of the satellite")
              axes2[3, 1].set(xlabel="Detector max count rate (hit/s)\nSideDetY")
              axes2[3, 2].set(xlabel="Detector max count rate (hit/s)\nLayer1")
              axes2[3, 3].set(xlabel="Detector max count rate (hit/s)\nLayer2")
              axes2[3, 4].set(xlabel="Detector max count rate (hit/s)\nCalorimeter")
              for itequad in range(len(axes2)):
                for itedet, ax in enumerate(axes2[itequad]):
                  for ite_sat_id in range(len(sat_id_list)):
                    sat_id = sat_id_list[ite_sat_id]
                    if ite_sat_id == max_sat_idx:
                      ax.scatter(sats_det_stats_shaped[itequad][itedet][ite_sat_id], sat_id, color="red")
                    else:
                      ax.scatter(sats_det_stats_shaped[itequad][itedet][ite_sat_id], sat_id, color="blue")
    fig, axes = plt.subplots(4, 5)
    fig.suptitle(f"Detectors max count rate histograms - {n_grb} brightest GRBs at peak")
    # ! ajouter bkg
    axes[0, 0].set(ylabel="Quad1\nNumber of detection")
    axes[1, 0].set(ylabel="Quad2\nNumber of detection")
    axes[2, 0].set(ylabel="Quad3\nNumber of detection")
    axes[3, 0].set(xlabel="Detector max count rate (hit/s)\nSideDetX", ylabel="Quad4\nNumber of detection")
    axes[3, 1].set(xlabel="Detector max count rate (hit/s)\nSideDetY")
    axes[3, 2].set(xlabel="Detector max count rate (hit/s)\nLayer1")
    axes[3, 3].set(xlabel="Detector max count rate (hit/s)\nLayer2")
    axes[3, 4].set(xlabel="Detector max count rate (hit/s)\nCalorimeter")
    for itequad in range(len(axes)):
      for itedet, ax in enumerate(axes[itequad]):
        ax.hist(combined_sats_det_stats_shaped[itequad][itedet], bins=30, color="blue")
        if n_grb > 20 :
          ax.set(yscale="log")

    plt.show()

  def bkg_det_stats(self):
    combined_sats_det_stats_bkg = None
    bkg_indexs = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            for sat in sim:
              if sat is not None:
                if sat.bkg_index not in bkg_indexs:
                  bkg_indexs.append(sat.bkg_index)

    for bkg_idx in bkg_indexs:
      bkg_stats = (self.bkgdata.bkgdf.com_det_stats.values[bkg_idx] + self.bkgdata.bkgdf.sin_det_stats.values[bkg_idx]).reshape(4, 5, 1) / self.bkgdata.sim_time
      if combined_sats_det_stats_bkg is None:
        combined_sats_det_stats_bkg = bkg_stats
        print(combined_sats_det_stats_bkg)
      else:
        combined_sats_det_stats_bkg = np.concatenate((combined_sats_det_stats_bkg, bkg_stats), axis=2)
        print(combined_sats_det_stats_bkg)

    fig, axes = plt.subplots(4, 5)
    axes[0, 0].set(ylabel="Quad1\nDetector count rate (hit/s)")
    axes[1, 0].set(ylabel="Quad2\nDetector count rate (hit/s)")
    axes[2, 0].set(ylabel="Quad3\nDetector count rate (hit/s)")
    axes[3, 0].set(xlabel="Time(s)\nSideDetX", ylabel="Quad4\nDetector count rate (hit/s)")
    axes[3, 1].set(xlabel="Time(s)\nSideDetY")
    axes[3, 2].set(xlabel="Time(s)\nLayer1")
    axes[3, 3].set(xlabel="Time(s)\nLayer2")
    axes[3, 4].set(xlabel="Time(s)\nCalorimeter")
    for itequad in range(len(axes)):
      for itedet, ax in enumerate(axes[itequad]):
        ax.hist(combined_sats_det_stats_bkg[itequad][itedet], bins=10, color="green")
    plt.show()
