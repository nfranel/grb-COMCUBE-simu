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

# Developped modules imports
from funcmod import *
from catalog import Catalog, SampleCatalog
from MBkgContainer import BkgContainer
from MmuSeffContainer import MuSeffContainer
from MAllSimData import AllSimData
from MLogData import LogData

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
    self.rest_cat_file = "./GBM/rest_frame_properties.txt"
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
    subprocess.call(f"make -f Makefile PRG=find_detector", shell=True)
    print("Compiling of the position finder finished")
    endtask("Step 1")

    # Setting the background files
    printcom("Step 2 - Extracting background data")
    init_time = time()
    self.bkgdata = BkgContainer(self.bkg_param, self.save_time, self.erg_cut)
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
      self.namelist = cat_data.df.name
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
    self.n_sim_simulated, self.n_sim_below_horizon, self.n_sim_in_radbelt, grb_names, grb_det_ites, sim_det_ites, sat_det_ites, suffix_ite = LogData(self.sim_prefix).detection_statistics(cat_data, False)
    endtask("Step 5", timevar=init_time)

    # Memory check for the class
    if memory_check:
      print("==================================== Memory check ====================================")
      print(f"  self.n_sim_simulated memory used : {asizeof.asizeof(self.n_sim_simulated)} bytes")
      print(f"  self.n_sim_below_horizon memory used : {asizeof.asizeof(self.n_sim_below_horizon)} bytes")
      print(f"  self.n_sim_in_radbelt memory used : {asizeof.asizeof(self.n_sim_in_radbelt)} bytes")
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

    if not os.path.exists(f"{self.sim_prefix.split('/sim/')[0]}/extracted-{self.erg_cut[0]}-{self.erg_cut[1]}"):
      os.mkdir(f"{self.sim_prefix.split('/sim/')[0]}/extracted-{self.erg_cut[0]}-{self.erg_cut[1]}")
    tobe_extracted, extracted_name, presence_list = self.filenames_creation(grb_names, grb_det_ites, sim_det_ites, sat_det_ites, suffix_ite)
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
        pool.starmap(save_grb_data, zip(tobe_extracted, extracted_name, repeat(self.sat_info), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.erg_cut), repeat(self.armcut), repeat(self.geometry)))
    elif type(parallel) is int:
      print(f"Parallel extraction of the data with {parallel} threads")
      with mp.Pool(parallel) as pool:
        pool.starmap(save_grb_data, zip(tobe_extracted, extracted_name, repeat(self.sat_info), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.erg_cut), repeat(self.armcut), repeat(self.geometry)))
    else:
      [save_grb_data(tobe_extracted[ext_ite], extracted_name[ext_ite], self.sat_info, self.bkgdata, self.muSeffdata, *self.options[:3]) for ext_ite in range(len(tobe_extracted))]
    endtask("Step 7", timevar=init_time)

    printcom("Step 8 - Loading log data and simulation statistics")
    init_time = time()
    # Reading the information from the extracted simulation files
    if parallel == 'all':
      print("Parallel extraction of the data with all threads")
      with mp.Pool() as pool:
        self.alldata = pool.starmap(AllSimData, zip(presence_list, range(self.n_source), repeat(cat_data), repeat(self.sat_info), repeat(self.sim_duration), repeat(self.options)))
    elif type(parallel) is int:
      print(f"Parallel extraction of the data with {parallel} threads")
      with mp.Pool(parallel) as pool:
        self.alldata = pool.starmap(AllSimData, zip(presence_list, range(self.n_source), repeat(cat_data), repeat(self.sat_info), repeat(self.sim_duration), repeat(self.options)))
    else:
      self.alldata = [AllSimData(presence_list[source_ite], source_ite, cat_data, self.sat_info, self.sim_duration, self.options) for source_ite in range(self.n_source)]
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
    for ite, grbname in enumerate(grb_names):
      temp_simfile = f"{self.sim_prefix}_{grbname}_sat{sat_det_ites[ite]}_{sim_det_ites[ite]:04d}_{suffix_ite[ite]}.inc1.id1.extracted.tra"
      tobe_ext.append(temp_simfile)
      temp_name = f"{self.sim_prefix.split('/sim/')[0]}/extracted-{self.erg_cut[0]}-{self.erg_cut[1]}/{self.sim_prefix.split('/sim/')[1]}_extracted{grbname}_sat{sat_det_ites[ite]}_{sim_det_ites[ite]:04d}_erg-{self.erg_cut[0]}-{self.erg_cut[1]}_arm-{self.armcut}.txt"
      ext_name.append(temp_name)
      pres_list[grb_det_ites[ite]][sim_det_ites[ite]][sat_det_ites[ite]] = temp_name
    return tobe_ext, ext_name, pres_list.tolist()
  #
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
            print("Removing 1 sat")
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
    # Search for an mdp limit :
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

  def count_triggers(self, const_index=0, graphs=False, lc_aligned=False):
    """
    Function to count and print the number of triggers using different criterions
    """
    print("================================================================================================")
    print(f"== Triggers according to GBM method with   {self.number_of_down_per_const[const_index]}   down satellite")
    print("================================================================================================")
    total_in_view = 0
    const_trigger_counter_4s = 0
    const_trigger_counter_3s = 0
    const_trigger_counter_2s = 0
    const_trigger_counter_1s = 0
    no_trig_name = []
    no_trig_duration = []
    no_trig_dec = []
    no_trig_e_fluence = []
    for source in self.alldata:
      if source is not None:
        for ite_sim, sim in enumerate(source):
          if sim is not None:
            total_in_view += 1
            if lc_aligned:
              list_snrs_lc_2s = []
              list_snrs_lc_3s = []
              list_snrs_lc_4s = []
              for sat in sim:
                if sat is not None:
                  # 2 sat trigger
                  if len(list_snrs_lc_2s) == 0:
                    list_snrs_lc_2s = sat.hits_snrs_over_lc(source.source_duration, nsat=2)
                  else:
                    temp_snrs_lc_2s = sat.hits_snrs_over_lc(source.source_duration, nsat=2)
                    for int_time_ite in range(len(list_snrs_lc_2s)):
                      list_snrs_lc_2s[int_time_ite] += temp_snrs_lc_2s[int_time_ite]
                  # 3 sat trigger
                  if len(list_snrs_lc_3s) == 0:
                    list_snrs_lc_3s = sat.hits_snrs_over_lc(source.source_duration, nsat=3)
                  else:
                    temp_snrs_lc_3s = sat.hits_snrs_over_lc(source.source_duration, nsat=3)
                    for int_time_ite in range(len(list_snrs_lc_3s)):
                      list_snrs_lc_3s[int_time_ite] += temp_snrs_lc_3s[int_time_ite]
                  # 4 sat trigger
                  if len(list_snrs_lc_4s) == 0:
                    list_snrs_lc_4s = sat.hits_snrs_over_lc(source.source_duration, nsat=4)
                  else:
                    temp_snrs_lc_4s = sat.hits_snrs_over_lc(source.source_duration, nsat=4)
                    for int_time_ite in range(len(list_snrs_lc_4s)):
                      list_snrs_lc_4s[int_time_ite] += temp_snrs_lc_4s[int_time_ite]
              if True in (np.concatenate(list_snrs_lc_2s) >= 3):
                const_trigger_counter_2s += 1
              if True in (np.concatenate(list_snrs_lc_3s) >= 3):
                const_trigger_counter_3s += 1
              if True in (np.concatenate(list_snrs_lc_4s) >= 3):
                const_trigger_counter_4s += 1
              if True in (sim.const_data[const_index].const_beneficial_trigger_1s >= 1):
                const_trigger_counter_1s += 1
            else:
              if sim.const_data[const_index] is not None:
                if True in (sim.const_data[const_index].const_beneficial_trigger_4s >= 4):
                  const_trigger_counter_4s += 1
                if True in (sim.const_data[const_index].const_beneficial_trigger_3s >= 3):
                  const_trigger_counter_3s += 1
                else:
                  no_trig_name.append(source.source_name)
                  no_trig_duration.append(source.source_duration)
                  no_trig_dec.append(sim.dec_world_frame)
                  no_trig_e_fluence.append(source.source_energy_fluence)
                  # if len(no_trig_name) <= 30 and graphs:
                  #   print("Not triggered : ", source.source_name, source.source_duration, sim.dec_world_frame, source.source_energy_fluence)
                if True in (sim.const_data[const_index].const_beneficial_trigger_2s >= 2):
                  const_trigger_counter_2s += 1
                if True in (sim.const_data[const_index].const_beneficial_trigger_1s >= 1):
                  const_trigger_counter_1s += 1

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


  # def count_triggers(self, const_index=0):
  #   """
  #   Function to count and print the number of triggers using different criterions
  #   """
  #   total_in_view = 0
  #   # Setting 1s mean triggers counter
  #   single_instant_trigger_by_const = 0
  #   single_instant_trigger_by_sat = 0
  #   single_instant_trigger_by_comparison = 0
  #   # Setting 1s peak triggers counter
  #   single_peak_trigger_by_const = 0
  #   single_peak_trigger_by_sat = 0
  #   single_peak_trigger_by_comparison = 0
  #   # Setting T90 mean triggers counter
  #   single_t90_trigger_by_const = 0
  #   single_t90_trigger_by_sat = 0
  #   single_t90_trigger_by_comparison = 0
  #
  #   for source in self.alldata:
  #     if source is not None:
  #       for sim in source:
  #         if sim is not None:
  #           total_in_view += 1
  #           #    Setting the trigger count to 0
  #           # Instantaneous trigger
  #           sat_instant_triggers = 0
  #           sat_reduced_instant_triggers = 0
  #           # Peak trigger
  #           sat_peak_triggers = 0
  #           sat_reduced_peak_triggers = 0
  #           # t90 trigger
  #           sat_t90_triggers = 0
  #           sat_reduced_t90_triggers = 0
  #           # Calculation for the individual sats
  #           for sat in sim:
  #             if sat is not None:
  #               if sat.snr_single >= self.snr_min:
  #                 sat_instant_triggers += 1
  #               if sat.snr_single >= self.snr_min - 2:
  #                 sat_reduced_instant_triggers += 1
  #               if source.best_fit_p_flux is None:
  #                 sat_peak_snr = sat.snr_single
  #               else:
  #                 sat_peak_snr = calc_snr(rescale_cr_to_GBM_pf(sat.single_cr, source.best_fit_mean_flux, source.best_fit_p_flux), sat.single_b_rate)
  #               if sat_peak_snr >= self.snr_min:
  #                 sat_peak_triggers += 1
  #               if sat_peak_snr >= self.snr_min - 2:
  #                 sat_reduced_peak_triggers += 1
  #               if sat.snr_single_t90 >= self.snr_min:
  #                 sat_t90_triggers += 1
  #               if sat.snr_single_t90 >= self.snr_min - 2:
  #                 sat_reduced_t90_triggers += 1
  #           # Calculation for the whole constellation
  #           if source.best_fit_p_flux is None:
  #             const_peak_snr = sim.const_data[const_index].snr_single
  #           else:
  #             const_peak_snr = calc_snr(rescale_cr_to_GBM_pf(sim.const_data[const_index].single_cr, source.best_fit_mean_flux, source.best_fit_p_flux), sim.const_data[const_index].single_b_rate)
  #           # 1s mean triggers
  #           if sim.const_data[const_index].snr_single >= self.snr_min:
  #             single_instant_trigger_by_const += 1
  #           if sat_instant_triggers >= 1:
  #             single_instant_trigger_by_sat += 1
  #           if sat_reduced_instant_triggers >= 3:
  #             single_instant_trigger_by_comparison += 1
  #           # 1s peak triggers
  #           if const_peak_snr >= self.snr_min:
  #             single_peak_trigger_by_const += 1
  #           if sat_peak_triggers >= 1:
  #             single_peak_trigger_by_sat += 1
  #           if sat_reduced_peak_triggers >= 3:
  #             single_peak_trigger_by_comparison += 1
  #           # T90 mean triggers
  #           if sim.const_data[const_index].snr_single_t90 >= self.snr_min:
  #             single_t90_trigger_by_const += 1
  #           if sat_t90_triggers >= 1:
  #             single_t90_trigger_by_sat += 1
  #           if sat_reduced_t90_triggers >= 3:
  #             single_t90_trigger_by_comparison += 1
  #
  #   print("The number of trigger for single events for the different technics are the following :")
  #   print(" == Integration time for the trigger : 1s, mean flux == ")
  #   print(f"   For a {self.snr_min} sigma trigger with the number of hits summed over the constellation :  {single_instant_trigger_by_const:.2f} triggers")
  #   # print(f"   For a {self.snr_min} sigma trigger on at least one of the satellites :                      {single_instant_trigger_by_sat:.2f} triggers")
  #   print(f"   For a {self.snr_min-2} sigma trigger in at least 3 satellites of the constellation :        {single_instant_trigger_by_comparison:.2f} triggers")
  #   # print(" == Integration time for the trigger : T90, mean flux == ")
  #   # print(f"   For a {self.snr_min} sigma trigger with the number of hits summed over the constellation : {single_t90_trigger_by_const} triggers")
  #   # print(f"   For a {self.snr_min} sigma trigger on at least one of the satellites : {single_t90_trigger_by_sat} triggers")
  #   # print(f"   For a {self.snr_min-2} sigma trigger in at least 3 satellites of the constellation : {single_t90_trigger_by_comparison} triggers")
  #   print("The number of trigger using GBM pflux for an energy range between 10keV and 1MeV are the following :")
  #   print(" == Integration time for the trigger : 1s, peak flux == ")
  #   print(f"   For a {self.snr_min} sigma trigger with the number of hits summed over the constellation :  {single_peak_trigger_by_const:.2f} triggers")
  #   # print(f"   For a {self.snr_min} sigma trigger on at least one of the satellites :                      {single_peak_trigger_by_sat:.2f} triggers")
  #   print(f"   For a {self.snr_min-2} sigma trigger in at least 3 satellites of the constellation :        {single_peak_trigger_by_comparison:.2f} triggers")
  #   print("=============================================")
  #   print(f" Over the {total_in_view} GRBs simulated in the constellation field of view")
  #
  #   print("================================================================================================")
  #   print("== Triggers according to GBM method")
  #   print("================================================================================================")
  #   # bin_widths = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 10]
  #   all_cat = Catalog(self.cat_file, self.sttype)
  #   total_in_view = 0
  #   bin_widths = [0.064, 0.256, 1.024]
  #   sat_trigger_counter = 0
  #   const_trigger_counter = 0
  #   for source in self.alldata:
  #     if source is not None:
  #       for ite_sim, sim in enumerate(source):
  #         if sim is not None:
  #           total_in_view += 1
  #           # Getting the snrs for all satellites and the constellation
  #           list_bins = [np.arange(0, source.source_duration + width, width) for width in bin_widths]
  #           centroid_bins = [(list_bin[1:] + list_bin[:-1]) / 2 for list_bin in list_bins]
  #           sat_snr_list = []
  #           sat_trigg = 0
  #           for sat_ite, sat in enumerate(sim):
  #             if sat is not None:
  #               if len(np.concatenate((sat.compton_time, sat.single_time))) == 0:
  #                 sat_trigg += 0
  #               else:
  #                 temp_hist = [np.histogram(np.concatenate((sat.compton_time, sat.single_time)), bins=list_bin)[0] for list_bin in list_bins]
  #                 arg_max_bin = [np.argmax(val_hist) for val_hist in temp_hist]
  #                 # for index1, arg_max1 in enumerate(arg_max_bin):
  #                 #   for index2, arg_max2 in enumerate(arg_max_bin[index1 + 1:]):
  #                 #     if not compatibility_test(centroid_bins[index1][arg_max1], bin_widths[index1], centroid_bins[index1 + 1 + index2][arg_max2], bin_widths[index1 + 1 + index2]):
  #                 # print(f"Incompatibility between bins {bin_widths[index1]} and {bin_widths[index2]} for {source.source_name}, sim {ite_sim} and sat {sat_ite}")
  #                 # print(f"     Centroids of the incompatible bins : {centroid_bins[index1][arg_max1]} and {centroid_bins[index1 + 1 + index2][arg_max2]}")
  #                 snr_list = [calc_snr(temp_hist[index][arg_max_bin[index]], (sat.single_b_rate + sat.compton_b_rate) * bin_widths[index]) for index in range(len(arg_max_bin))]
  #                 sat_snr_list.append(snr_list)
  #                 if max(snr_list) > 3:
  #                   sat_trigg += 1
  #           if sim.const_data[const_index] is not None:
  #             if len(np.concatenate((sim.const_data[const_index].compton_time, sim.const_data[const_index].single_time))) == 0:
  #               const_snr = [0]
  #             else:
  #               temp_hist = [np.histogram(np.concatenate((sim.const_data[const_index].compton_time, sim.const_data[const_index].single_time)), bins=list_bin)[0] for list_bin in list_bins]
  #               arg_max_bin = [np.argmax(val_hist) for val_hist in temp_hist]
  #               # for index1, arg_max1 in enumerate(arg_max_bin):
  #               #   for index2, arg_max2 in enumerate(arg_max_bin[index1 + 1:]):
  #               #     if not compatibility_test(centroid_bins[index1][arg_max1], bin_widths[index1], centroid_bins[index1 + 1 + index2][arg_max2], bin_widths[index1 + 1 + index2]):
  #               # print(f"Incompatibility between bins {bin_widths[index1]} and {bin_widths[index2]} for {source.source_name}, sim {ite_sim} and constellation")
  #               # print(f"     Centroids of the incompatible bins : {centroid_bins[index1][arg_max1]} and {centroid_bins[index1 + 1 + index2][arg_max2]}")
  #               const_snr = [calc_snr(temp_hist[index][arg_max_bin[index]], (sim.const_data[const_index].single_b_rate + sim.const_data[const_index].compton_b_rate) * bin_widths[index]) for index in range(len(arg_max_bin))]
  #           else:
  #             const_snr = [0]
  #           if sat_trigg >= 4:
  #             sat_trigger_counter += 1
  #           if max(const_snr) >= 6:
  #             const_trigger_counter += 1
  #           else:
  #             for ite, name in enumerate(all_cat.name):
  #               if name == source.source_name:
  #                 ener_fluence = float(all_cat.fluence[ite])
  #             print(max(const_snr), source.source_duration, sim.dec_world_frame, ener_fluence)
  #   print(
  #     f"   For a 6 sigma trigger with the number of hits summed over the constellation :  {const_trigger_counter:.2f} triggers")
  #   print(
  #     f"   For a 3 sigma trigger in at least 4 satellites of the constellation :        {sat_trigger_counter:.2f} triggers")
  #   print("=============================================")
  #   print(f" Over the {total_in_view} GRBs simulated in the constellation field of view")

  def fov_const(self, num_val=500, show=True, save=False):
    """
    Plots a map of the sensibility over the sky for number of sat in sight, single events and compton events
    :param num_val: number of value to
    """
    phi_world = np.linspace(0, 360, num_val, endpoint=False)
    # theta will be converted in sat coord with grb_decra_worldf2satf, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, 180, num_val)
    detection = np.zeros((self.n_sat, num_val, num_val))
    detection_compton = np.zeros((self.n_sat, num_val, num_val))
    detection_single = np.zeros((self.n_sat, num_val, num_val))

    # for ite in range(self.n_sat):
    #   detection_pola[ite] = np.array([[eff_area_compton_func(grb_decra_worldf2satf(theta, phi, self.sat_info[ite][0], self.sat_info[ite][1])[0], self.sat_info[ite][2], func_type="cos") for phi in phi_world] for theta in theta_world])
    #   detection_spectro[ite] = np.array([[eff_area_single_func(grb_decra_worldf2satf(theta, phi, self.sat_info[ite][0], self.sat_info[ite][1])[0], self.sat_info[ite][2], func_type="data") for phi in phi_world] for theta in theta_world])

    for ite, info_sat in enumerate(self.sat_info):
      for ite_theta, theta in enumerate(theta_world):
        for ite_phi, phi in enumerate(phi_world):
          detection_compton[ite][ite_theta][ite_phi], detection_single[ite][ite_theta][ite_phi], detection[ite][ite_theta][ite_phi] = eff_area_func(theta, phi, info_sat, self.muSeffdata)
    detec_sum = np.sum(detection, axis=0)
    detec_sum_compton = np.sum(detection_compton, axis=0)
    detec_sum_single = np.sum(detection_single, axis=0)

    phi_plot, theta_plot = np.meshgrid(phi_world, theta_world)
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
    # Map for number of satellite in sight
    ##################################################################################################################
    levels = range(detec_min, detec_max + 1, max(1, int(detec_max + 1 - detec_min) / 15))

    # fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig1, ax1 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    ax1.set_global()
    ax1.coastlines()
    h1 = ax1.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_det)
    ax1.axis('scaled')
    ax1.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
    cbar = fig1.colorbar(h1, ticks=levels)
    cbar.set_label("Number of satellite in sight", rotation=270, labelpad=20)
    if save:
      fig1.savefig(f"{self.result_prefix}_n_sight")
    if show:
      plt.show()

    # plt.subplot(projection="mollweide")
    # h2 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_det)
    # plt.axis('scaled')
    # plt.xlabel("Right ascention (rad)")
    # plt.ylabel("Declination (rad)")
    # cbar = plt.colorbar(ticks=levels)
    # cbar.set_label("Number of satellite in sight", rotation=270, labelpad=20)
    # if save:
    #   plt.savefig(f"{self.result_prefix}_n_sight_proj")
    # if show:
    #   plt.show()

    ##################################################################################################################
    # Map of constellation's compton effective area
    ##################################################################################################################
    levels_compton = range(detec_min_compton, detec_max_compton + 1, max(1, int(detec_max_compton + 1 - detec_min_compton) / 15))

    # fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    ax2.set_global()
    ax2.coastlines()
    h3 = ax2.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_compton, cmap=cmap_compton)
    ax2.axis('scaled')
    ax2.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
    cbar = fig2.colorbar(h3, ticks=levels_compton)
    cbar.set_label("Effective area at for compton events (cm²)", rotation=270, labelpad=20)
    if save:
      fig2.savefig(f"{self.result_prefix}_compton_seff")
    if show:
      plt.show()

    # plt.subplot(projection="mollweide")
    # h4 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_compton, cmap=cmap_compton)
    # plt.axis('scaled')
    # plt.xlabel("Right ascention (rad)")
    # plt.ylabel("Declination (rad)")
    # cbar = plt.colorbar(ticks=levels_compton)
    # cbar.set_label("Effective area at for compton events (cm²)", rotation=270, labelpad=20)
    # if save:
    #   plt.savefig(f"{self.result_prefix}_compton_seff_proj")
    # if show:
    #   plt.show()

    ##################################################################################################################
    # Map of constellation's compton effective area
    ##################################################################################################################
    levels_single = range(detec_min_single, detec_max_single + 1, max(1, int(detec_max_single + 1 - detec_min_single) / 15))

    # fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    fig3, ax3 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    ax3.set_global()
    ax3.coastlines()
    h5 = ax3.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_single, cmap=cmap_single)
    ax3.axis('scaled')
    ax3.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
    cbar = fig3.colorbar(h5, ticks=levels_single)
    cbar.set_label("Effective area for single events (cm²)", rotation=270, labelpad=20)
    if save:
      fig3.savefig(f"{self.result_prefix}_single_seff")
    if show:
      plt.show()

    # plt.subplot(projection="mollweide")
    # h6 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_single, cmap=cmap_single)
    # plt.axis('scaled')
    # plt.xlabel("Right ascention (rad)")
    # plt.ylabel("Declination (rad)")
    # cbar = plt.colorbar(ticks=levels_single)
    # cbar.set_label("Effective area for single events (cm²)", rotation=270, labelpad=20)
    # if save:
    #   plt.savefig(f"{self.result_prefix}_single_seff")
    # if show:
    #   plt.show()

    print(f"The mean number of satellite in sight is :       {np.mean(detec_sum):.4f} satellites")
    print(f"The mean effective area for compton events is :  {np.mean(detec_sum_compton):.4f} cm²")
    print(f"The mean effective area for single events is :   {np.mean(detec_sum_single):.4f} cm²")

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

  def brightest_det_stats(self, n_grb):
    lst_pflux = []
    for source in self.alldata:
      if source is not None:
        if source.best_fit_p_flux is not None:
          lst_pflux.append(source.best_fit_p_flux)
        else:
          lst_pflux.append(0)
    lst_pflux = np.array(lst_pflux, dtype=np.float32)
    idxs = np.argsort(lst_pflux)[:n_grb]

    for idx in idxs:
      if self.alldata[idx] is not None:
        for sim in self.alldata[idx]:
          if sim is not None:
            max_sat_idx = np.argmax(np.array([sat.compton * 2 + sat.single if sat is not None else 0 for sat in sim]))
            sats_det_stats = []
            for ite_sat, sat in enumerate(sim):
              if sat is not None:
                if ite_sat == max_sat_idx:
                  sats_det_stats.append(sat.detector_statistics(self.bkgdata[sat.bkg_index], self.bkgdata.sim_time, self.alldata[idx].source_duration, self.alldata[idx].source_name, show=True))
                else:
                  sats_det_stats.append(sat.detector_statistics(self.bkgdata[sat.bkg_index], self.bkgdata.sim_time, self.alldata[idx].source_duration, self.alldata[idx].source_name, show=False))
            sats_det_stats_shaped = np.transpose(np.array(sats_det_stats), (1, 2, 0))
            print(sats_det_stats)
            print(sats_det_stats_shaped)

            # ! test the shape change, the detector stat gathering and the graphs
            # ! ajouter bkg
            fig, axes = plt.subplots(4, 5)
            fig.suptitle(f"Detectors max count rate - {n_grb} brightest GRBs at peak")
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
                ax.hist(sats_det_stats_shaped[itequad][itedet], color="blue")
            plt.show()

  # todo change it
  # def hits_energy_histogram(self, num_grb, num_sim, energy_type="both", selected_sat="const", n_bins=30,
  #                           x_scale='log', y_scale='linear'):
  #   """
  #   Plots the energy spectrum for a detection by 1 sat or by the constellation
  #   :param num_grb: index of the GRB
  #   :param num_sim: index of the simulation (its number)
  #   :param energy_type: "both", "compton" or "single", to select which event is considered
  #   :param selected_sat: int or string, which sat is selected, if "const" the constellation is selected
  #   :param n_bins: number of bins in the histogram
  #   :param x_scale: scale for x-axis
  #   :param y_scale: scale for y-axis
  #   """
  #   hits_energy = []
  #   if selected_sat == "const":
  #     file_string = f"{self.namelist[num_grb]}, simulation {num_sim} and the whole constellation"
  #   else:
  #     file_string = f"{self.namelist[num_grb]}, simulation {num_sim} and the satellite {selected_sat}"
  #   if energy_type == "compton":
  #     title = f"Energy distribution of compton events for the source {file_string}"
  #   elif energy_type == "single":
  #     title = f"Energy distribution of single events for the source {file_string}"
  #   elif energy_type == "both":
  #     title = f"Energy distribution of compton and single events for the source {file_string}"
  #   else:
  #     print("Choose a correct type of event for enery histograms : both(default), compton, single ")
  #     return "hits_energy_histogram error"
  #
  #   if self.alldata[num_grb] is not None:
  #     if self.alldata[num_grb][num_sim] is not None:
  #       if type(selected_sat) is int:
  #         if self.alldata[num_grb][num_sim][selected_sat] is not None:
  #           if energy_type == "compton":
  #             hits_energy = self.alldata[num_grb][num_sim][selected_sat].compton_ener
  #           elif energy_type == "single":
  #             hits_energy = self.alldata[num_grb][num_sim][selected_sat].single_ener
  #           elif energy_type == "both":
  #             hits_energy = np.concatenate((self.alldata[num_grb][num_sim][selected_sat].compton_ener, self.alldata[num_grb][num_sim][selected_sat].single_ener))
  #         else:
  #           print(
  #             f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]} on the selected sat : {selected_sat}, no histogram drawn")
  #       elif selected_sat == "const":
  #         if energy_type == "compton":
  #           hits_energy = self.alldata[num_grb][num_sim].const_data.compton_ener
  #         elif energy_type == "single":
  #           hits_energy = self.alldata[num_grb][num_sim].const_data.single_ener
  #         elif energy_type == "both":
  #           hits_energy = np.concatenate((self.alldata[num_grb][num_sim].const_data.compton_ener,
  #                                         self.alldata[num_grb][num_sim].const_data.single_ener))
  #     else:
  #       print(f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]}, no histogram drawn")
  #   else:
  #     print(f"No detection for this source : {self.namelist[num_grb]}, no histogram drawn")
  #
  #   if x_scale == "log":
  #     if min(hits_energy) < 1:
  #       inf_limit = int(np.log10(min(hits_energy))) - 1
  #     else:
  #       inf_limit = int(np.log10(min(hits_energy)))
  #     if max(hits_energy) > 1:
  #       sup_limit = int(np.log10(max(hits_energy))) + 1
  #     else:
  #       sup_limit = int(np.log10(max(hits_energy)))
  #     hist_bins = np.logspace(inf_limit, sup_limit, n_bins)
  #   else:
  #     hist_bins = n_bins
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   ax1.hist(hits_energy, bins=hist_bins, cumulative=0, histtype="step")
  #   ax1.set(xlabel="Energy (keV)", ylabel="Number of photon detected", xscale=x_scale, yscale=y_scale)
  #   plt.show()

  # todo change it
  # def arm_histogram(self, num_grb, num_sim, selected_sat="const", n_bins=30, arm_lim=0.8,
  #                   x_scale='linear', y_scale='linear'):
  #   """
  #
  #   """
  #   arm_values = []
  #   if self.alldata[num_grb] is not None:
  #     if self.alldata[num_grb][num_sim] is not None:
  #       if type(selected_sat) is int:
  #         if self.alldata[num_grb][num_sim][selected_sat] is not None:
  #           arm_values = self.alldata[num_grb][num_sim][selected_sat].arm
  #         else:
  #           print(
  #             f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]} on the selected sat : {selected_sat}, no histogram drawn")
  #           return
  #       elif selected_sat == "const":
  #         arm_values = self.alldata[num_grb][num_sim].const_data.arm
  #     else:
  #       print(f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]}, no histogram drawn")
  #       return
  #   else:
  #     print(f"No detection for this source : {self.namelist[num_grb]}, no histogram drawn")
  #     return
  #
  #   arm_threshold = np.sort(arm_values)[int(len(arm_values) * arm_lim - 1)]
  #
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle("ARM distribution of photons for an event")
  #   ax1.hist(arm_values, bins=n_bins, cumulative=0, histtype="step")
  #   ax1.axvline(arm_threshold, color="black", label=f"{arm_lim * 100}% values limit = {arm_threshold}")
  #   ax1.set(xlabel="Angular Resolution Measurement (°)", ylabel="Number of photon detected", xscale=x_scale,
  #           yscale=y_scale)
  #   ax1.legend()
  #   plt.show()

  # todo change it all
  # def peak_flux_distri(self, snr_type="compton", selected_sat="const", snr_min=None, n_bins=30, x_scale='log', y_scale="log"):
  #   """
  #
  #   """
  #   if snr_min is None:
  #     snr_min = self.snr_min
  #   hist_pflux = []
  #   for source in self.alldata:
  #     if source is not None and source.best_fit_p_flux is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if selected_sat == "const":
  #             if snr_type == "compton":
  #               if sim.const_data.snr_compton_t90 >= snr_min:
  #                 hist_pflux.append(source.best_fit_p_flux)
  #             elif snr_type == "single":
  #               if sim.const_data.snr_single_t90 >= snr_min:
  #                 hist_pflux.append(source.best_fit_p_flux)
  #           else:
  #             if snr_type == "compton":
  #               if sim[selected_sat] is not None:
  #                 if sim[selected_sat].snr_compton_t90 is not None:
  #                   if sim[selected_sat].snr_compton_t90 >= snr_min:
  #                     hist_pflux.append(source.best_fit_p_flux)
  #             elif snr_type == "single":
  #               if sim[selected_sat] is not None:
  #                 if sim[selected_sat].snr_single_t90 is not None:
  #                   if sim[selected_sat].snr_single_t90 >= snr_min:
  #                     hist_pflux.append(source.best_fit_p_flux)
  #
  #   if x_scale == "log":
  #     if np.min(hist_pflux) < 1:
  #       inf_limit = int(np.log10(min(hist_pflux))) - 1
  #     else:
  #       inf_limit = int(np.log10(min(hist_pflux)))
  #     if np.max(hist_pflux) > 1:
  #       sup_limit = int(np.log10(max(hist_pflux))) + 1
  #     else:
  #       sup_limit = int(np.log10(max(hist_pflux)))
  #     hist_bins = np.logspace(inf_limit, sup_limit, n_bins)
  #   else:
  #     hist_bins = n_bins
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle("Peak flux distribution of detected GRB")
  #   ax1.hist(hist_pflux, bins=hist_bins, cumulative=False, histtype="step", weights=[self.weights/(4*np.pi)] * len(hist_pflux))
  #   ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Number of detection per year per steradian", xscale=x_scale, yscale=y_scale)
  #   # ax1.legend()
  #   plt.show()
  #
  # def det_proba_vs_pflux(self, selected_sat="const", x_scale='log', y_scale='linear'):
  #   """
  #   sat contains either the number of the satellite selected or "const"
  #   """
  #   p_flux_list = []
  #   det_prob_fov_list = []
  #   det_prob_sky_list = []
  #   for source in self.alldata:
  #     if source is not None and source.best_fit_p_flux is not None:
  #       p_flux_list.append(source.best_fit_p_flux)
  #       if selected_sat == "const":
  #         det_prob_fov_list.append(source.const_single_proba_detec_fov)
  #         det_prob_sky_list.append(source.const_single_proba_detec_sky)
  #       else:
  #         det_prob_fov_list.append(source.proba_detec_fov[selected_sat])
  #         det_prob_sky_list.append(source.proba_detec_sky[selected_sat])
  #
  #   distrib, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
  #   distrib.suptitle("Detection probability vs peak flux of detected GRB - GRB in the whole sky (left) and only in the FoV (right)")
  #   ax1.scatter(p_flux_list, det_prob_sky_list, s=2, label='Detection probability over the whole sky')
  #   ax2.scatter(p_flux_list, det_prob_fov_list, s=2, label='Detection probability over the field of view')
  #
  #   ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability", xscale=x_scale, yscale=y_scale)
  #   ax1.legend()
  #   ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability", xscale=x_scale, yscale=y_scale)
  #   ax2.legend()
  #   plt.show()
  #
  # def compton_im_proba_vs_pflux(self, selected_sat="const", x_scale='log', y_scale='linear'):
  #   """
  #   sat contains either the number of the satellite selected or "const"
  #   """
  #   p_flux_list = []
  #   comp_im_prob_fov_list = []
  #   comp_im_prob_sky_list = []
  #   for source in self.alldata:
  #     if source is not None and source.best_fit_p_flux is not None:
  #       p_flux_list.append(source.best_fit_p_flux)
  #       if selected_sat == "const":
  #         comp_im_prob_fov_list.append(source.const_proba_compton_image_fov)
  #         comp_im_prob_sky_list.append(source.const_proba_compton_image_sky)
  #       else:
  #         comp_im_prob_fov_list.append(source.proba_compton_image_fov[selected_sat])
  #         comp_im_prob_sky_list.append(source.proba_compton_image_sky[selected_sat])
  #   distrib, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
  #   distrib.suptitle(
  #     "Compton Image probability vs peak flux of detected GRB - GRB in the whole sky (left) and only in the FoV (right)")
  #   ax1.scatter(p_flux_list, comp_im_prob_sky_list, s=2, label='Compton image probability over the whole sky')
  #   ax2.scatter(p_flux_list, comp_im_prob_fov_list, s=2, label='Compton image probability over the field of view')
  #
  #   ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability", xscale=x_scale, yscale=y_scale)
  #   ax1.legend()
  #   ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability", xscale=x_scale, yscale=y_scale)
  #   ax2.legend()
  #   plt.show()
  #
  # def mu100_distri(self, selected_sat="const", n_bins=30, x_scale='linear', y_scale="log"):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     title = "mu100 distribution of GRBs detected by the whole constellation"
  #   else:
  #     title = f"mu100 distribution of GRBs detected by the satellite {selected_sat}"
  #
  #   mu_100_list = []
  #   for source in self.alldata:
  #     if source is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if selected_sat == "const":
  #             if sim.const_data.mu100 is not None:
  #               mu_100_list.append(sim.const_data.mu100)
  #           else:
  #             if sim[selected_sat] is not None:
  #               if sim[selected_sat].mu100 is not None:
  #                 mu_100_list.append(sim[selected_sat].mu100)
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   ax1.hist(mu_100_list, bins=n_bins, cumulative=0, histtype="step", weights=[self.weights] * len(mu_100_list))
  #   ax1.set(xlabel="mu100 (dimensionless)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale)
  #   plt.show()
  #
  # def mu100_vs_angle(self, selected_sat=0, x_scale='linear', y_scale="linear"):
  #   """
  #
  #   """
  #   title = f"mu100 vs detection angle of GRBs detected by the satellite {selected_sat}"
  #   # Faire de façon à ce que tous les satellites détectés soient pris en compte aussi ~ const option
  #   mu_100_list = []
  #   angle_list = []
  #   for source in self.alldata:
  #     if source is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if sim[selected_sat] is not None:
  #             if sim[selected_sat].mu100 is not None:
  #               mu_100_list.append(sim[selected_sat].mu100)
  #               angle_list.append(sim[selected_sat].dec_sat_frame)
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   ax1.scatter(mu_100_list, angle_list, s=2)
  #   ax1.set(xlabel="mu100 (dimensionless)", ylabel="Detection angle (°)", xlim=(0, 1), ylim=(115, 0), xscale=x_scale, yscale=y_scale)
  #   plt.show()
  #
  # def pa_distribution(self, selected_sat="const", n_bins=30, x_scale='linear', y_scale="log"):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     title = "Polarization angle distribution of GRBs detected by the whole constellation"
  #   else:
  #     title = f"Polarization angle distribution of GRBs detected by the satellite {selected_sat}"
  #
  #   pa_list = []
  #   for source in self.alldata:
  #     if source is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if selected_sat == "const":
  #             if sim.const_data.pa is not None:
  #               pa_list.append(sim.const_data.pa)
  #           else:
  #             if sim[selected_sat] is not None:
  #               if sim[selected_sat].pa is not None:
  #                 pa_list.append(sim[selected_sat].pa)
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   ax1.hist(pa_list, bins=n_bins, cumulative=0, histtype="step", weights=[self.weights] * len(pa_list))
  #   ax1.set(xlabel="Polarization angle (°)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale)
  #   plt.show()
  #
  # def mdp_vs_fluence(self, selected_sat="const", mdp_threshold=1, x_scale='log', y_scale='linear'):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     title = "MDP as a function of fluence of GRBs detected by the whole constellation"
  #   else:
  #     title = f"MDP as a function of fluence of GRBs detected by the satellite {selected_sat}"
  #
  #   if not self.cat_file == "None":
  #     mdp_list = []
  #     fluence_list = []
  #     mdp_count = 0
  #     no_detec_fluence = []
  #     for source_ite, source in enumerate(self.alldata):
  #       if source is not None:
  #         for sim in source:
  #           if sim is not None:
  #             if selected_sat == "const":
  #               if sim.const_data.mdp is not None:
  #                 if sim.const_data.mdp <= mdp_threshold:
  #                   mdp_list.append(sim.const_data.mdp * 100)
  #                   fluence_list.append(source.source_fluence)
  #                 else:
  #                   no_detec_fluence.append(source.source_fluence)
  #               mdp_count += 1
  #             else:
  #               if sim[selected_sat] is not None:
  #                 if sim[selected_sat].mdp is not None:
  #                   if sim[selected_sat].mdp <= mdp_threshold:
  #                     mdp_list.append(sim[selected_sat].mdp * 100)
  #                     fluence_list.append(source.source_fluence)
  #                   else:
  #                     no_detec_fluence.append(source.source_fluence)
  #                 mdp_count += 1
  #
  #     distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #     distrib.suptitle(title)
  #     for ite_val, val in enumerate(np.unique(no_detec_fluence)):
  #       if ite_val == 0:
  #         ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
  #       else:
  #         ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
  #     ax1.scatter(fluence_list, mdp_list, s=3,
  #                 label=f'Detected GRB polarization \nRatio of detectable polarization : {len(mdp_list) / mdp_count}')
  #     ax1.set(xlabel="fluence (erg.cm-2)", ylabel="MDP (%)", xscale=x_scale, yscale=y_scale,
  #             xlim=(10 ** (int(np.log10(np.min(fluence_list))) - 1), 10 ** (int(np.log10(np.max(fluence_list))) + 1)))
  #     ax1.legend()
  #     plt.show()
  #
  # def mdp_vs_pflux(self, selected_sat="const", mdp_threshold=1, x_scale='log', y_scale='linear'):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     title = "MDP as a function of flux at peak of GRBs detected by the whole constellation"
  #   else:
  #     title = f"MDP as a function of flux at peak of GRBs detected by the satellite {selected_sat}"
  #
  #   mdp_list = []
  #   flux_list = []
  #   mdp_count = 0
  #   no_detec_flux = []
  #   for source_ite, source in enumerate(self.alldata):
  #     if source is not None and source.best_fit_p_flux is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if selected_sat == "const":
  #             if sim.const_data.mdp is not None:
  #               if sim.const_data.mdp <= mdp_threshold:
  #                 mdp_list.append(sim.const_data.mdp * 100)
  #                 flux_list.append(source.best_fit_p_flux)
  #               else:
  #                 no_detec_flux.append(source.best_fit_p_flux)
  #             mdp_count += 1
  #           else:
  #             if sim[selected_sat] is not None:
  #               if sim[selected_sat].mdp is not None:
  #                 if sim[selected_sat].mdp <= mdp_threshold:
  #                   mdp_list.append(sim[selected_sat].mdp * 100)
  #                   flux_list.append(source.best_fit_p_flux)
  #                 else:
  #                   no_detec_flux.append(source.best_fit_p_flux)
  #               mdp_count += 1
  #
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   for ite_val, val in enumerate(np.unique(no_detec_flux)):
  #     if ite_val == 0:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
  #     else:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
  #   ax1.scatter(flux_list, mdp_list, s=3,
  #               label=f'Detected GRB polarization \nRatio of detectable polarization : {len(mdp_list) / mdp_count}')
  #   ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="MDP (%)", xscale=x_scale, yscale=y_scale,
  #           xlim=(10 ** (int(np.log10(np.min(flux_list))) - 1), 10 ** (int(np.log10(np.max(flux_list))) + 1)))
  #   ax1.legend()
  #   plt.show()
  #
  # def mdp_vs_detection_angle(self, selected_sat=0, mdp_threshold=1, x_scale='linear', y_scale='linear'):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     title = "MDP as a function of detection angle of GRBs detected by the whole constellation"
  #   else:
  #     title = f"MDP as a function of detection angle of GRBs detected by the satellite {selected_sat}"
  #
  #   mdp_list = []
  #   angle_list = []
  #   mdp_count = 0
  #   no_detec_angle = []
  #   for source_ite, source in enumerate(self.alldata):
  #     if source is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if sim[selected_sat] is not None:
  #             if sim[selected_sat].mdp is not None:
  #               if sim[selected_sat].mdp <= mdp_threshold:
  #                 mdp_list.append(sim[selected_sat].mdp * 100)
  #                 angle_list.append(sim[selected_sat].dec_sat_frame)
  #               else:
  #                 no_detec_angle.append(sim[selected_sat].dec_sat_frame)
  #               mdp_count += 1
  #
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   for ite_val, val in enumerate(np.unique(no_detec_angle)):
  #     if ite_val == 0:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
  #     else:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
  #   ax1.scatter(angle_list, mdp_list, s=3,
  #               label=f'Detected GRB polarization \nRatio of detectable polarization : {len(mdp_list) / mdp_count}')
  #   ax1.set(xlabel="Angle (°)", ylabel="MDP (%)", xscale=x_scale, yscale=y_scale,
  #           xlim=(0, 180))
  #   ax1.legend()
  #   plt.show()
  #
  # def snr_vs_fluence(self, snr_type="compton", selected_sat="const", snr_threshold=5, x_scale='log', y_scale='log'):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     file_string = "the whole constellation"
  #   else:
  #     file_string = f"the satellite {selected_sat}"
  #   if snr_type == "compton":
  #     title = f"SNR of compton events as a function of fluence of GRBs detected by {file_string}"
  #   elif snr_type == "single":
  #     title = f"SNR of single events as a function of fluence of GRBs detected by {file_string}"
  #   else:
  #     print("Choose a correct type of snr : compton(default), single")
  #     return "snr_vs_fluence error"
  #
  #   if not self.cat_file == "None":
  #     snr_list = []
  #     fluence_list = []
  #     snr_count = 0
  #     no_detec_fluence = []
  #     for source_ite, source in enumerate(self.alldata):
  #       if source is not None:
  #         for sim in source:
  #           if sim is not None:
  #             if selected_sat == "const":
  #               if snr_type == "compton":
  #                 if sim.const_data.snr_compton_t90 >= snr_threshold:
  #                   snr_list.append(sim.const_data.snr_compton_t90)
  #                   fluence_list.append(source.source_fluence)
  #                 else:
  #                   no_detec_fluence.append(source.source_fluence)
  #               elif snr_type == "single":
  #                 if sim.const_data.snr_single_t90 >= snr_threshold:
  #                   snr_list.append(sim.const_data.snr_single_t90)
  #                   fluence_list.append(source.source_fluence)
  #                 else:
  #                   no_detec_fluence.append(source.source_fluence)
  #               snr_count += 1
  #             else:
  #               if snr_type == "compton":
  #                 if sim[selected_sat] is not None:
  #                   if sim[selected_sat].snr_compton_t90 is not None:
  #                     if sim[selected_sat].snr_compton_t90 >= snr_threshold:
  #                       snr_list.append(sim[selected_sat].snr_compton_t90)
  #                       fluence_list.append(source.source_fluence)
  #                     else:
  #                       no_detec_fluence.append(source.source_fluence)
  #                     snr_count += 1
  #               elif snr_type == "single":
  #                 if sim[selected_sat] is not None:
  #                   if sim[selected_sat].snr_single_t90 is not None:
  #                     if sim[selected_sat].snr_single_t90 >= snr_threshold:
  #                       snr_list.append(sim[selected_sat].snr_single_t90)
  #                       fluence_list.append(source.source_fluence)
  #                     else:
  #                       no_detec_fluence.append(source.source_fluence)
  #                     snr_count += 1
  #
  #     distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #     distrib.suptitle(title)
  #     for ite_val, val in enumerate(np.unique(no_detec_fluence)):
  #       if ite_val == 0:
  #         ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
  #       else:
  #         ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
  #     ax1.scatter(fluence_list, snr_list, s=3,
  #                 label=f'Detected GRB SNR \nRatio of detectable GRB : {len(snr_list) / snr_count}')
  #     ax1.set(xlabel="Fluence (erg.cm-2)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale,
  #             xlim=(10 ** (int(np.log10(np.min(fluence_list))) - 1), 10 ** (int(np.log10(np.max(fluence_list))) + 1)))
  #     ax1.legend()
  #     plt.show()
  #
  # def snr_vs_pflux(self, snr_type="compton", selected_sat="const", snr_threshold=5, x_scale='log', y_scale='log'):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     file_string = "the whole constellation"
  #   else:
  #     file_string = f"the satellite {selected_sat}"
  #   if snr_type == "compton":
  #     title = f"SNR of compton events as a function of peak flux of GRBs detected by {file_string}"
  #   elif snr_type == "single":
  #     title = f"SNR of single events as a function of peak flux of GRBs detected by {file_string}"
  #   else:
  #     print("Choose a correct type of snr : compton(default), single")
  #     return "snr_vs_fluence error"
  #
  #   snr_list = []
  #   flux_list = []
  #   snr_count = 0
  #   no_detec_flux = []
  #   for source_ite, source in enumerate(self.alldata):
  #     if source is not None and source.best_fit_p_flux is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if selected_sat == "const":
  #             if snr_type == "compton":
  #               if sim.const_data.snr_compton_t90 >= snr_threshold:
  #                 snr_list.append(sim.const_data.snr_compton_t90)
  #                 flux_list.append(self.alldata[source_ite].best_fit_p_flux)
  #               else:
  #                 no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
  #             elif snr_type == "single":
  #               if sim.const_data.snr_single_t90 >= snr_threshold:
  #                 snr_list.append(sim.const_data.snr_single_t90)
  #                 flux_list.append(self.alldata[source_ite].best_fit_p_flux)
  #               else:
  #                 no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
  #             snr_count += 1
  #           else:
  #             if snr_type == "compton":
  #               if sim[selected_sat] is not None:
  #                 if sim[selected_sat].snr_compton_t90 is not None:
  #                   if sim[selected_sat].snr_compton_t90 >= snr_threshold:
  #                     snr_list.append(sim[selected_sat].snr_compton_t90)
  #                     flux_list.append(self.alldata[source_ite].best_fit_p_flux)
  #                   else:
  #                     no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
  #                   snr_count += 1
  #             elif snr_type == "single":
  #               if sim[selected_sat] is not None:
  #                 if sim[selected_sat].snr_single_t90 is not None:
  #                   if sim[selected_sat].snr_single_t90 >= snr_threshold:
  #                     snr_list.append(sim[selected_sat].snr_single_t90)
  #                     flux_list.append(self.alldata[source_ite].best_fit_p_flux)
  #                   else:
  #                     no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
  #                   snr_count += 1
  #
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   for ite_val, val in enumerate(np.unique(no_detec_flux)):
  #     if ite_val == 0:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
  #     else:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
  #   ax1.scatter(flux_list, snr_list, s=3,
  #               label=f'Detected GRB SNR \nRatio of detectable GRB : {len(snr_list) / snr_count}')
  #   ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale,
  #           xlim=(10 ** (int(np.log10(np.min(flux_list))) - 1), 10 ** (int(np.log10(np.max(flux_list))) + 1)))
  #   ax1.legend()
  #   plt.show()
  #
  # def snr_vs_detection_angle(self, snr_type="compton", selected_sat=0, snr_threshold=5, x_scale='linear', y_scale='log'):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     file_string = "the whole constellation"
  #   else:
  #     file_string = f"the satellite {selected_sat}"
  #   if snr_type == "compton":
  #     title = f"SNR of compton events as a function of detection angle of GRBs detected by {file_string}"
  #   elif snr_type == "single":
  #     title = f"SNR of single events as a function of detection angle of GRBs detected by {file_string}"
  #   else:
  #     print("Choose a correct type of snr : compton(default), single")
  #     return "snr_vs_fluence error"
  #
  #   snr_list = []
  #   angle_list = []
  #   snr_count = 0
  #   no_detec_angle = []
  #   for source_ite, source in enumerate(self.alldata):
  #     if source is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if snr_type == "compton":
  #             if sim[selected_sat] is not None:
  #               if sim[selected_sat].snr_compton_t90 is not None:
  #                 if sim[selected_sat].snr_compton_t90 >= snr_threshold:
  #                   snr_list.append(sim[selected_sat].snr_compton_t90)
  #                   angle_list.append(sim[selected_sat].dec_sat_frame)
  #                 else:
  #                   no_detec_angle.append(sim[selected_sat].dec_sat_frame)
  #                 snr_count += 1
  #           elif snr_type == "single":
  #             if sim[selected_sat] is not None:
  #               if sim[selected_sat].snr_single_t90 is not None:
  #                 if sim[selected_sat].snr_single_t90 >= snr_threshold:
  #                   snr_list.append(sim[selected_sat].snr_single_t90)
  #                   angle_list.append(sim[selected_sat].dec_sat_frame)
  #                 else:
  #                   no_detec_angle.append(sim[selected_sat].dec_sat_frame)
  #                 snr_count += 1
  #
  #   distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
  #   distrib.suptitle(title)
  #   for ite_val, val in enumerate(np.unique(no_detec_angle)):
  #     if ite_val == 0:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
  #     else:
  #       ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
  #   ax1.scatter(angle_list, snr_list, s=3,
  #               label=f'Detected GRB SNR \nRatio of detectable GRB : {len(snr_list) / snr_count}')
  #   ax1.set(xlabel="Detection angle (°)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale,
  #           xlim=(0, 180))
  #   ax1.legend()
  #   plt.show()
  #
  # def mdp_vs_snr(self, selected_sat="const", snr_threshold=5, mdp_threshold=1, print_rejected=False,
  #                x_scale='log', y_scale='linear'):
  #   """
  #
  #   """
  #   if selected_sat == "const":
  #     title = "SNR as a function of MDP of GRBs detected by the whole constellation"
  #   else:
  #     title = f"SNR as a function of MDP of GRBs detected by the satellite {selected_sat}"
  #
  #   mdp_list = []
  #   snr_list = []
  #   count = 0
  #   no_detec = [[], []]
  #   for source_ite, source in enumerate(self.alldata):
  #     if source is not None:
  #       for sim in source:
  #         if sim is not None:
  #           if selected_sat == "const":
  #             if sim.const_data.snr_compton_t90 >= snr_threshold and sim.const_data.mdp <= mdp_threshold:
  #               mdp_list.append(sim.const_data.mdp * 100)
  #               snr_list.append(sim.const_data.snr_compton_t90)
  #             else:
  #               no_detec[0].append(sim.const_data.mdp * 100)
  #               no_detec[1].append(sim.const_data.snr_compton_t90)
  #             count += 1
  #           else:
  #             if sim[selected_sat] is not None:
  #               if sim[selected_sat].snr_compton_t90 is not None:
  #                 if sim[selected_sat].snr_compton_t90 >= snr_threshold and sim[selected_sat].mdp <= mdp_threshold:
  #                   mdp_list.append(sim[selected_sat].mdp * 100)
  #                   snr_list.append(sim[selected_sat].snr_compton_t90)
  #                 else:
  #                   no_detec[0].append(sim[selected_sat].mdp * 100)
  #                   no_detec[1].append(sim[selected_sat].snr_compton_t90)
  #                 count += 1
  #
  #     distrib, ax1 = plt.subplots(1, 1, figsize=(10, 6))
  #     distrib.suptitle(title)
  #     if print_rejected:
  #       ax1.scatter(no_detec[0], no_detec[1], label="Markers for rejected GRB")
  #     ax1.scatter(mdp_list, snr_list, s=3,
  #                 label=f'Detected GRB (Both with SNR and MDP) \nRatio of detectable GRB : {len(snr_list) / count}')
  #     ax1.set(xlabel="Fluence (erg.cm-2)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale)
  #     ax1.legend()
  #     plt.show()
