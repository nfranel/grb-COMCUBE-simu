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
# Developped modules imports
from funcmod import *
from catalog import Catalog
from MBkgContainer import BkgContainer
from MmuSeffContainer import MuSeffContainer
from MAllSimData import AllSimData
from MLogData import LogData

# Ploting adjustments
mpl.use('Qt5Agg')
# plt.rcParams.update({'font.size': 20})


class AllSourceData:
  """
  Class containing all the data for a full set of trafiles
  """

  def __init__(self, grb_param, bkg_param, muSeff_param, erg_cut=(100, 460), armcut=180, polarigram_bins="fixed", parallel=False):
    """
    Initiate the class AllData using
    - bkg_prefix : str, the prefix for background files
    - param_file : str, the path to the parameter file (.par) used for the simulation
    - erg_cut    : tuple of len 2, the lower and uppuer bounds of the energy window considered

    Extract from the parameters and from the files the information needed for the analysis
    Makes some basic tests on filenames to reduce the risk of unseen errors

    FAIRE OPTION AVEC PARAM FILE QUI EST UNE LISTE (evite d'avoir a faire un .par si on veut juste étudier 1 seule simu)
    Ceci est le cas de base pour les simulations, le modifier pour permettre des sources moins habituelles
    """
    # General parameters
    self.grb_param = grb_param
    self.bkg_param = bkg_param
    self.muSeff_param = muSeff_param
    self.erg_cut = erg_cut
    self.armcut = armcut
    # Different kinds of bins can be made :
    if polarigram_bins in ["fixed", "limited", "optimized"]:
      self.polarigram_bins = polarigram_bins
    else:
      print("Warning : wrong option for the polarigram bins, it should be fixed (default), limited or optimized. Hence the option has been set to default value.")
      self.polarigram_bins = "fixed"
    # Setup of some options
    self.save_pos = True
    self.save_time = True
    self.init_correction = False
    self.snr_min = 5
    # self.options = [self.save_pos, self.save_time, self.polarigram_bins, self.armcut, self.init_correction,self.erg_cut]
    self.options = [self.erg_cut, self.armcut, self.save_pos, self.save_time, self.init_correction, self.polarigram_bins]

# self.bkg_sim_duration = 3600
    # Parameters extracted from parfile
    self.geometry, self.revan_file, self.mimrec_file, self.spectra_path, self.cat_file, self.source_file, self.sim_prefix, self.sttype, self.n_sim, self.sim_duration, self.position_allowed_sim, self.sat_info = read_grbpar(self.grb_param)
    self.n_sat = len(self.sat_info)

    # Setting the background files
    self.bkgdata = BkgContainer(self.bkg_param, self.save_pos, self.save_time, self.erg_cut)

    # Setting the background files
    self.muSeffdata = MuSeffContainer(self.muSeff_param, self.erg_cut, self.armcut)

    # Log information
    # log = LogData("/pdisk/ESA/test--400km--0-0-0--27sat")
    self.n_sim_simulated, self.n_sim_below_horizon, self.n_sim_in_radbelt = LogData(self.sim_prefix.split("/sim/")[0]).detection_statistics()

    # Setting the background rate detected by each satellite
    # for sat_ite in range(len(self.sat_info)):
    #   for count_rates in closest_bkg_rate(self.sat_info[sat_ite][1], self.sat_info[sat_ite][0], self.bkgdata):
    #     self.sat_info[sat_ite].append(count_rates)
    # TODO set the bkg information to each sim file because the sats are moving now

    # Setting the catalog and the attributes associated
    if self.cat_file == "None":
      cat_data = self.extract_sources(self.sim_prefix)
      self.namelist = cat_data[0]
      self.n_source = len(self.namelist)
    else:
      cat_data = Catalog(self.cat_file, self.sttype)
      self.namelist = cat_data.name
      self.n_source = len(self.namelist)

    # Extracting the informations from the simulation files
    if parallel == 'all':
      print("Parallel extraction of the data with all threads")
      with mp.Pool() as pool:
        self.alldata = pool.starmap(AllSimData, zip(repeat(self.sim_prefix), range(self.n_source), repeat(cat_data), repeat(self.n_sim), repeat(self.sat_info), repeat(self.sim_duration), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.options)))
    elif type(parallel) is int:
      print(f"Parallel extraction of the data with {parallel} threads")
      with mp.Pool(parallel) as pool:
        self.alldata = pool.starmap(AllSimData, zip(repeat(self.sim_prefix), range(self.n_source), repeat(cat_data), repeat(self.n_sim), repeat(self.sat_info), repeat(self.sim_duration), repeat(self.bkgdata), repeat(self.muSeffdata), repeat(self.options)))
    else:
      self.alldata = [AllSimData(self.sim_prefix, source_ite, cat_data, self.n_sim, self.sat_info, self.sim_duration, self.bkgdata, self.muSeffdata, self.options) for source_ite in range(self.n_source)]

    # Setting some informations used for obtaining the GRB count rates
    self.cat_duration = 10
    # self.com_duty = 1
    self.com_duty = self.n_sim_simulated / (self.n_sim_simulated + self.n_sim_in_radbelt)
    self.gbm_duty = 0.85
    ### Implementer une maniere automatique de calculer le fov de comcube
    self.com_fov = 1  # kept as 1 because GRBs simulated accross all sky and not considered if behind the earth
    self.gbm_fov = (1 - np.cos(np.deg2rad(horizonAngle(565)))) / 2
    self.weights = 1 / self.n_sim / self.cat_duration * self.com_duty / self.gbm_duty * self.com_fov / self.gbm_fov

  @staticmethod
  def get_keys():
    print("======================================================================")
    print("    Files and paths")
    print(" background files prefix :            .bkg_prefix")
    print(" Parameter file used for simulation : .param_file")
    print(" Simulated data prefix :              .sim_prefix")
    print(" Source file path :                   .source_file")
    print(" Revan cfg file path :                .revan_file")
    print(" Geometry file path :                 .geometry")
    print(" Catalog file path :                  .cat_file")
    print(" Path of spectra :                    .spectra_path")
    print("======================================================================")
    print("    Simulation parameters")
    print(" Type of simulation from parfile :         .sim_type")  # Might be usefull to handle different types of sim
    print(" Instrument fiel from parfile    :         .instrument")
    print(" Mode used to handle catalog information : .mode")
    print(" Formated str to extract catalog sources : .sttype")  # Might put in an other field ?
    print(" Area of the sky allowed for simulations : .position_allowed_sim")
    print("======================================================================")
    print("    Data analysis options")
    print(" Energy window considered for the analysis :           .erg_cut")
    print(" Data extraction options :                             .options")
    print("   [save_pos, save_time, corr, erg_cut]")
    print("    save_pos : to get another fiels from trafiles")
    print("    save_time : to handle the new field with a specific function")
    print("    corr : to correct the polarization angle")
    print(" Whether or not bkg simulated with the source :         .source_with_bkg")
    print("======================================================================")
    print("    Data and simulation information")
    print(" Information on satellites' position :   .sat_info")
    print(" Number of satellites :                  .n_sat")
    print(" Number of simulation performed :        .n_sim")
    print(" Duration of simulations :               .sim_duration")
    print(" List of source names :                  .namelist")
    print(" Number of sources simulated :           .n_source")
    print(" Data extracted from simulation files :  .alldata")
    print("======================================================================")
    print("    Methods")
    print("======================================================================")

  def extract_sources(self, prefix, duration=None):
    """

    """
    if duration is None:
      if self.sim_duration.isdigit():
        duration = float(self.sim_duration)
      elif self.sim_duration == "t90":
        duration = None
        print("Warning : impossible to load the t90 as sim duration is no catalog is given.")
      else:
        duration = None
        print("Warning : unusual sim duration, please check the parameter file.")

    flist = subprocess.getoutput("ls {}_*".format(prefix)).split("\n")
    source_names = []
    if len(flist) >= 1 and not flist[0].startswith("ls: cannot access"):
      temp_sourcelist = []
      for file in flist:
        temp_sourcelist.append(file.split("_")[1])
      source_names = list(set(temp_sourcelist))
    return [source_names, [duration] * len(source_names)]

  def azi_angle_corr(self):
    """

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

    """
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            for sat in sim:
              if sat is not None:
                sat.anticorr()

  def analyze(self, fit_bounds=None, const_analysis=True):
    """
    Proceed to the analysis of polarigrams for all satellites and constellation (unless specified) for all data
    """
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim_ite, sim in enumerate(source):
          if sim is not None:
            sim.analyze(source.source_duration, source.source_fluence, const_analysis)
            # if source.source_fluence is None:
            #   sim.analyze(source.source_duration, source.source_fluence, self.source_with_bkg, fit_bounds, const_analysis)
            # else:
            #   sim.analyze(source.source_duration, source.source_fluence, self.source_with_bkg, fit_bounds, const_analysis)
        source.set_probabilities(n_sat=self.n_sat, snr_min=self.snr_min, n_image_min=50)

  def make_const(self, const=None):
    """
    This function is used to combine results from different satellites
    Results are then stored in the key const_data
    The polarigrams have to be corrected to combine the polarigrams
    """
    if not self.init_correction:
      self.azi_angle_corr()
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            sim.make_const(self.options, const=const)
    if not self.init_correction:
      self.azi_angle_anticorr()

  def verif_const(self, const=None):
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim_ite, sim in enumerate(source):
          if sim is not None:
            sim.verif_const(message=f"for source {source_ite} and sim {sim_ite}", const=const)

  def effective_area(self, sat=0):
    """
    sat is the number of the satellite considered
    This method is supposed to be working with a set of 40 satellites with a lot of simulations
    The results obtained with this method are meaningful only is there is no background simulated
    """
    if self.source_with_bkg:
      print(
        "WARNING : The source has been simulated with a background, the calculation has not been done as this would lead to biased results")
    else:
      list_dec = []
      list_s_eff_compton = []
      list_s_eff_single = []
      list_fluence = []
      for source in self.alldata:
        if source is not None:
          temp_dec = []
          temp_s_eff_compton = []
          temp_s_eff_single = []
          for num_sim, sim in enumerate(source):
            if sim is not None:
              if sim[sat] is None:
                print(
                  f"The satellite {sat} selected didn't detect the source '{source.source_name}' for the simulation number {num_sim}.")
              else:
                temp_dec.append(sim[sat].dec_sat_frame)
                temp_s_eff_compton.append(sim[sat].s_eff_compton)
                temp_s_eff_single.append(sim[sat].s_eff_single)
          list_dec.append(temp_dec)
          list_s_eff_compton.append(temp_s_eff_compton)
          list_s_eff_single.append(temp_s_eff_single)
          list_fluence.append(source.source_fluence)

      figure, ax = plt.subplots(2, 2, figsize=(16, 12))
      figure.suptitle("Effective area as a function of detection angle")
      for graph in range(4):
        for ite in range(graph * 10, min(graph * 10 + 10, len(list_dec))):
          ax[int(graph / 2)][graph % 2].scatter(list_dec[ite], list_s_eff_compton[ite],
                                                label=f"Fluence : {np.around(list_fluence[ite], decimals=1)} ph/cm²")
        ax[int(graph / 2)][graph % 2].set(xlabel="GRB zenith angle (rad)",
                                          ylabel="Effective area for polarimetry (cm²)")  # , yscale="linear")
        ax[int(graph / 2)][graph % 2].legend()
      plt.show()

  def source_search(self, source_name, verbose=True):
    """
    Search among the sources simulated if one has the correct name
    returns the position of the source(s) in the list and displays other information unless specified if it's there
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
      return
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
    return

  def fov_const(self, num_val=500, mode="polarization", show=True, save=False):
    """
    Plots a map of the sensibility (s_eff_compton) over the sky
    Mode is the mode used to obtain the sensibility :
      Polarization gives the sensibility to polarization
      Spectrometry gives the sensibility to spectrometry (capacity of detection)
    """
    phi_world = np.linspace(0, 360, num_val)
    # theta will be converted in sat coord with grb_decra_worldf2satf, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, 180, num_val)
    detection_pola = np.zeros((self.n_sat, num_val, num_val))
    detection_spectro = np.zeros((self.n_sat, num_val, num_val))

    for ite in range(self.n_sat):
      detection_pola[ite] = np.array([[eff_area_compton_func(grb_decra_worldf2satf(theta, phi, self.sat_info[ite][0],
                                                                                   self.sat_info[ite][1])[0], self.sat_info[ite][2], func_type="cos") for phi in phi_world] for theta in theta_world])
      detection_spectro[ite] = np.array([[eff_area_single_func(grb_decra_worldf2satf(theta, phi, self.sat_info[ite][0],
                                                                                     self.sat_info[ite][1])[0], self.sat_info[ite][2], func_type="data") for phi in phi_world] for theta in theta_world])

    detec_sum_pola = np.sum(detection_pola, axis=0)
    detec_sum_spectro = np.sum(detection_spectro, axis=0)

    phi_plot, theta_plot = np.meshgrid(phi_world, theta_world)
    detec_min_pola = int(np.min(detec_sum_pola))
    detec_max_pola = int(np.max(detec_sum_pola))
    detec_min_spectro = int(np.min(detec_sum_spectro))
    detec_max_spectro = int(np.max(detec_sum_spectro))
    cmap_pola = mpl.cm.Greens_r
    cmap_spectro = mpl.cm.Oranges_r

    # Eff_area plots for polarimetry
    # levels_pola = range(int(detec_min_pola / 2) * 2, detec_max_pola + 1)
    levels_pola = range(int(detec_min_pola), int(detec_max_pola) + 1, int((int(detec_max_pola) + 1 - int(detec_min_pola)) / 15))

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area at for polarisation (cm²)", rotation=270, labelpad=20)
    plt.savefig("figtest")
    if save:
      plt.savefig("eff_area_noproj_pola")
    if show:
      plt.show()

    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area for polarisation (cm²)", rotation=270, labelpad=20)
    if save:
      plt.savefig("eff_area_proj_pola")
    if show:
      plt.show()

    # Eff_area plots for spectroscopy
    # levels_spectro = range(int(detec_min_spectro / 2) * 2, detec_max_spectro + 1)
    levels_spectro = range(int(detec_min_spectro), int(detec_max_spectro) + 1,
                           int((int(detec_max_spectro) + 1 - int(detec_min_spectro)) / 15))

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    if save:
      plt.savefig("eff_area_noproj_spectro")
    if show:
      plt.show()

    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    if save:
      plt.savefig("eff_area_proj_spectro")
    if show:
      plt.show()

    print(f"La surface efficace moyenne pour la polarisation est de {np.mean(np.mean(detec_sum_pola, axis=1))} cm²")
    print(f"La surface efficace moyenne pour la spectrométrie est de {np.mean(np.mean(detec_sum_spectro, axis=1))} cm²")

  def count_triggers(self):
    """

    """
    total_in_view = 0
    # Setting 1s mean triggers counter
    single_instant_trigger_by_const = 0
    single_instant_trigger_by_sat = 0
    single_instant_trigger_by_comparison = 0
    # Setting 1s peak triggers counter
    single_peak_trigger_by_const = 0
    single_peak_trigger_by_sat = 0
    single_peak_trigger_by_comparison = 0
    # Setting T90 mean triggers counter
    single_t90_trigger_by_const = 0
    single_t90_trigger_by_sat = 0
    single_t90_trigger_by_comparison = 0

    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            total_in_view += 1
            #    Setting the trigger count to 0
            # Instantaneous trigger
            sat_instant_triggers = 0
            sat_reduced_instant_triggers = 0
            # Peak trigger
            sat_peak_triggers = 0
            sat_reduced_peak_triggers = 0
            # t90 trigger
            sat_t90_triggers = 0
            sat_reduced_t90_triggers = 0
            # Calculation for the individual sats
            for sat in sim:
              if sat is not None:
                if sat.snr_single >= self.snr_min:
                  sat_instant_triggers += 1
                if sat.snr_single >= self.snr_min - 2:
                  sat_reduced_instant_triggers += 1
                sat_peak_snr = SNR(rescale_cr_to_GBM_pf(sat.single_cr, source.best_fit_mean_flux, source.best_fit_p_flux), sat.single_b_rate)
                print("rescaled cr : ", rescale_cr_to_GBM_pf(sat.single_cr, source.best_fit_mean_flux, source.best_fit_p_flux))
                print("initial cr : ", sat.single_cr)
                print("peak flux : ", source.best_fit_p_flux)
                print("mean flux : ", source.best_fit_mean_flux)
                print("mean flux in ergcut :", source.source_fluence / source.source_duration)
                print("b_rate : ", sat.single_b_rate)
                print("snr : ", sat_peak_snr)
                if sat_peak_snr >= self.snr_min:
                  sat_peak_triggers += 1
                if sat_peak_snr >= self.snr_min - 2:
                  sat_reduced_peak_triggers += 1
                if sat.snr_single_t90 >= self.snr_min:
                  sat_t90_triggers += 1
                if sat.snr_single_t90 >= self.snr_min - 2:
                  sat_reduced_t90_triggers += 1
            # Calculation for the whole constellation
            const_peak_snr = SNR(rescale_cr_to_GBM_pf(sim.const_data.single_cr, source.best_fit_mean_flux, source.best_fit_p_flux), sim.const_data.single_b_rate)
            print()
            print("rescaled cr : ", rescale_cr_to_GBM_pf(sim.const_data.single_cr, source.best_fit_mean_flux, source.best_fit_p_flux))
            print("initial cr : ", sim.const_data.single_cr)
            print("peak flux : ", source.best_fit_p_flux)
            print("reduced peak flux : ", source.best_fit_p_flux * source.source_fluence / source.source_duration / source.best_fit_mean_flux)
            print("mean flux : ", source.best_fit_mean_flux)
            print("mean flux in ergcut :", source.source_fluence / source.source_duration)
            print("b_rate : ", sim.const_data.single_b_rate)
            print("         snr peak : ", const_peak_snr)
            print("         snr mean : ", sim.const_data.snr_single)
            print()
            # Summing for simulated values
            # 1s mean triggers
            if sim.const_data.snr_single >= self.snr_min:
              single_instant_trigger_by_const += 1
            if sat_instant_triggers >= 1:
              single_instant_trigger_by_sat += 1
            if sat_reduced_instant_triggers >= 3:
              single_instant_trigger_by_comparison += 1
            # 1s peak triggers
            if const_peak_snr >= self.snr_min:
              single_peak_trigger_by_const += 1
            if sat_peak_triggers >= 1:
              single_peak_trigger_by_sat += 1
            if sat_reduced_peak_triggers >= 3:
              single_peak_trigger_by_comparison += 1
            # T90 mean triggers
            if sim.const_data.snr_single_t90 >= self.snr_min:
              single_t90_trigger_by_const += 1
            if sat_t90_triggers >= 1:
              single_t90_trigger_by_sat += 1
            if sat_reduced_t90_triggers >= 3:
              single_t90_trigger_by_comparison += 1

    print("The number of trigger for single events for the different technics are the following :")
    print(" == Integration time for the trigger : 1s, mean flux == ")
    print(f"   For a {self.snr_min} sigma trigger with the number of hits summed over the constellation : {single_instant_trigger_by_const} triggers")
    print(f"   For a {self.snr_min} sigma trigger on at least one of the satellites : {single_instant_trigger_by_sat} triggers")
    print(f"   For a {self.snr_min-2} sigma trigger in at least 3 satellites of the constellation : {single_instant_trigger_by_comparison} triggers")
    print(" == Integration time for the trigger : T90, mean flux == ")
    print(f"   For a {self.snr_min} sigma trigger with the number of hits summed over the constellation : {single_t90_trigger_by_const} triggers")
    print(f"   For a {self.snr_min} sigma trigger on at least one of the satellites : {single_t90_trigger_by_sat} triggers")
    print(f"   For a {self.snr_min-2} sigma trigger in at least 3 satellites of the constellation : {single_t90_trigger_by_comparison} triggers")
    print("The number of trigger using GBM pflux for an energy range between 10keV and 1MeV are the following :")
    print(" == Integration time for the trigger : 1s, peak flux == ")
    print(f"   For a {self.snr_min} sigma trigger with the number of hits summed over the constellation : {single_peak_trigger_by_const} triggers")
    print(f"   For a {self.snr_min} sigma trigger on at least one of the satellites : {single_peak_trigger_by_sat} triggers")
    print(f"   For a {self.snr_min-2} sigma trigger in at least 3 satellites of the constellation : {single_peak_trigger_by_comparison} triggers")
    print("=============================================")
    print(f" Over the {total_in_view} GRBs simulated in the constellation field of view")

  def grb_map_plot(self, mode="no_cm"):
    """
    Display the catalog GRBs position in the sky using the corresponding function in catalogext
    """
    cat_data = Catalog(self.cat_file, self.sttype)
    cat_data.grb_map_plot(mode)

  def spectral_information(self):
    """
    Displays the spectral information of the GRBs including the proportion of different best fit models and the
    corresponding parameters
    """
    cat_data = Catalog(self.cat_file, self.sttype)
    cat_data.spectral_information()

  def mdp_histogram(self, selected_sat="const", mdp_threshold=1, cumul=1, n_bins=30, x_scale='linear', y_scale="log"):
    """
    Display and histogram representing the number of grb of a certain mdp per year
    """
    if self.cat_file.endswith("longGBM.txt"):
      grb_type = "lGRB"
    elif self.cat_file.endswith("shortGRB.txt"):
      grb_type = "sGRB"
    else:
      grb_type = "undefined source"
    number_detected = 0
    mdp_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if type(selected_sat) is int:
              if sim[selected_sat] is not None:
                number_detected += 1
                if sim[selected_sat].mdp is not None:
                  if sim[selected_sat].mdp <= mdp_threshold:
                    mdp_list.append(sim[selected_sat].mdp * 100)
            elif selected_sat == "const":
              if sim.const_data is not None:
                number_detected += 1
                if sim.const_data.mdp is not None:
                  if sim.const_data.mdp <= mdp_threshold:
                    mdp_list.append(sim.const_data.mdp * 100)
    fig, ax = plt.subplots(1, 1)
    ax.hist(mdp_list, bins=n_bins, cumulative=cumul, histtype="step", weights=[self.weights] * len(mdp_list),
            label=f"Number of GRBs with MDP < {mdp_threshold * 100}% : {len(mdp_list)} over {number_detected} detections")
    if cumul == 1:
      ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Cumulative distribution of the MDP - {grb_type}")
    elif cumul == 0:
      ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Distribution of the MDP - {grb_type}")
    elif cumul == -1:
      ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
             title=f"Inverse cumulative distribution of the MDP - {grb_type}")
    ax.legend(loc='upper left')
    ax.grid(axis='both')
    plt.show()

  def snr_histogram(self, snr_type="compton", selected_sat="const", cumul=0, n_bins=30, x_scale="log", y_scale="log"):
    """
    Display and histogram representing the number of grb that have at least a certain snr per year
    """
    if self.cat_file.endswith("longGBM.txt"):
      grb_type = "lGRB"
    elif self.cat_file.endswith("shortGRB.txt"):
      grb_type = "sGRB"
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
            if selected_sat == "const":
              if snr_type == "compton":
                snr_list.append(sim.const_data.snr_compton_t90)
              elif snr_type == "single":
                snr_list.append(sim.const_data.snr_single_t90)
            else:
              if snr_type == "compton":
                snr_list.append(sim[selected_sat].snr_compton_t90)
              elif snr_type == "single":
                snr_list.append(sim[selected_sat].snr_single_t90)

    fig, ax = plt.subplots(1, 1)
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

  def hits_energy_histogram(self, num_grb, num_sim, energy_type="both", selected_sat="const", n_bins=30,
                            x_scale='log', y_scale='linear'):
    """

    """
    hits_energy = []
    if selected_sat == "const":
      file_string = f"{self.namelist[num_grb]}, simulation {num_sim} and the whole constellation"
    else:
      file_string = f"{self.namelist[num_grb]}, simulation {num_sim} and the satellite {selected_sat}"
    if energy_type == "compton":
      title = f"Energy distribution of compton events for the source {file_string}"
    elif energy_type == "single":
      title = f"Energy distribution of single events for the source {file_string}"
    elif energy_type == "both":
      title = f"Energy distribution of compton and single events for the source {file_string}"
    else:
      print("Choose a correct type of event for enery histograms : both(default), compton, single ")
      return "hits_energy_histogram error"

    if self.alldata[num_grb] is not None:
      if self.alldata[num_grb][num_sim] is not None:
        if type(selected_sat) is int:
          if self.alldata[num_grb][num_sim][selected_sat] is not None:
            if energy_type == "compton":
              hits_energy = self.alldata[num_grb][num_sim][selected_sat].compton_ener
            elif energy_type == "single":
              hits_energy = self.alldata[num_grb][num_sim][selected_sat].single_ener
            elif energy_type == "both":
              hits_energy = np.concatenate((self.alldata[num_grb][num_sim][selected_sat].compton_ener, self.alldata[num_grb][num_sim][selected_sat].single_ener))
          else:
            print(
              f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]} on the selected sat : {selected_sat}, no histogram drawn")
        elif selected_sat == "const":
          if energy_type == "compton":
            hits_energy = self.alldata[num_grb][num_sim].const_data.compton_ener
          elif energy_type == "single":
            hits_energy = self.alldata[num_grb][num_sim].const_data.single_ener
          elif energy_type == "both":
            hits_energy = np.concatenate((self.alldata[num_grb][num_sim].const_data.compton_ener,
                                          self.alldata[num_grb][num_sim].const_data.single_ener))
      else:
        print(f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]}, no histogram drawn")
    else:
      print(f"No detection for this source : {self.namelist[num_grb]}, no histogram drawn")

    if x_scale == "log":
      if min(hits_energy) < 1:
        inf_limit = int(np.log10(min(hits_energy))) - 1
      else:
        inf_limit = int(np.log10(min(hits_energy)))
      if max(hits_energy) > 1:
        sup_limit = int(np.log10(max(hits_energy))) + 1
      else:
        sup_limit = int(np.log10(max(hits_energy)))
      hist_bins = np.logspace(inf_limit, sup_limit, n_bins)
    else:
      hist_bins = n_bins
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    ax1.hist(hits_energy, bins=hist_bins, cumulative=0, histtype="step")
    ax1.set(xlabel="Energy (keV)", ylabel="Number of photon detected", xscale=x_scale, yscale=y_scale)
    plt.show()

  def arm_histogram(self, num_grb, num_sim, selected_sat="const", n_bins=30, arm_lim=0.8,
                    x_scale='linear', y_scale='linear'):
    """

    """
    arm_values = []
    if self.alldata[num_grb] is not None:
      if self.alldata[num_grb][num_sim] is not None:
        if type(selected_sat) is int:
          if self.alldata[num_grb][num_sim][selected_sat] is not None:
            arm_values = self.alldata[num_grb][num_sim][selected_sat].arm
          else:
            print(
              f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]} on the selected sat : {selected_sat}, no histogram drawn")
            return
        elif selected_sat == "const":
          arm_values = self.alldata[num_grb][num_sim].const_data.arm
      else:
        print(f"No detection for the simulation {num_sim} for the source {self.namelist[num_grb]}, no histogram drawn")
        return
    else:
      print(f"No detection for this source : {self.namelist[num_grb]}, no histogram drawn")
      return

    arm_threshold = np.sort(arm_values)[int(len(arm_values) * arm_lim - 1)]

    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle("ARM distribution of photons for an event")
    ax1.hist(arm_values, bins=n_bins, cumulative=0, histtype="step")
    ax1.axvline(arm_threshold, color="black", label=f"{arm_lim * 100}% values limit = {arm_threshold}")
    ax1.set(xlabel="Angular Resolution Measurement (°)", ylabel="Number of photon detected", xscale=x_scale,
            yscale=y_scale)
    ax1.legend()
    plt.show()

  def peak_flux_distri(self, snr_type="compton", selected_sat="const", snr_min=None, n_bins=30, x_scale='log', y_scale="log"):
    """

    """
    if snr_min is None:
      snr_min = self.snr_min
    hist_pflux = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if snr_type == "compton":
                if sim.const_data.snr_compton_t90 >= snr_min:
                  hist_pflux.append(source.best_fit_p_flux)
              elif snr_type == "single":
                if sim.const_data.snr_single_t90 >= snr_min:
                  hist_pflux.append(source.best_fit_p_flux)
            else:
              if snr_type == "compton":
                if sim[selected_sat] is not None:
                  if sim[selected_sat].snr_compton_t90 is not None:
                    if sim[selected_sat].snr_compton_t90 >= snr_min:
                      hist_pflux.append(source.best_fit_p_flux)
              elif snr_type == "single":
                if sim[selected_sat] is not None:
                  if sim[selected_sat].snr_single_t90 is not None:
                    if sim[selected_sat].snr_single_t90 >= snr_min:
                      hist_pflux.append(source.best_fit_p_flux)

    if x_scale == "log":
      if np.min(hist_pflux) < 1:
        inf_limit = int(np.log10(min(hist_pflux))) - 1
      else:
        inf_limit = int(np.log10(min(hist_pflux)))
      if np.max(hist_pflux) > 1:
        sup_limit = int(np.log10(max(hist_pflux))) + 1
      else:
        sup_limit = int(np.log10(max(hist_pflux)))
      hist_bins = np.logspace(inf_limit, sup_limit, n_bins)
    else:
      hist_bins = n_bins
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle("Peak flux distribution of detected long GRB")
    ax1.hist(hist_pflux, bins=hist_bins, cumulative=False, histtype="step", weights=[self.weights/(4*np.pi)] * len(hist_pflux))
    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Number of detection per year per steradian", xscale=x_scale, yscale=y_scale)
    # ax1.legend()
    plt.show()

  def det_proba_vs_pflux(self, selected_sat="const", x_scale='log', y_scale='linear'):
    """
    sat contains either the number of the satellite selected or "const"
    """
    p_flux_list = []
    det_prob_fov_list = []
    det_prob_sky_list = []
    for source in self.alldata:
      if source is not None:
        p_flux_list.append(source.best_fit_p_flux)
        if selected_sat == "const":
          det_prob_fov_list.append(source.const_single_proba_detec_fov)
          det_prob_sky_list.append(source.const_single_proba_detec_sky)
        else:
          det_prob_fov_list.append(source.proba_detec_fov[selected_sat])
          det_prob_sky_list.append(source.proba_detec_sky[selected_sat])

    distrib, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    distrib.suptitle("Detection probability vs peak flux of detected long GRB - GRB in the whole sky (left) and only in the FoV (right)")
    ax1.scatter(p_flux_list, det_prob_sky_list, s=2, label='Detection probability over the whole sky')
    ax2.scatter(p_flux_list, det_prob_fov_list, s=2, label='Detection probability over the field of view')

    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability", xscale=x_scale, yscale=y_scale)
    ax1.legend()
    ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability", xscale=x_scale, yscale=y_scale)
    ax2.legend()
    plt.show()

  def compton_im_proba_vs_pflux(self, selected_sat="const", x_scale='log', y_scale='linear'):
    """
    sat contains either the number of the satellite selected or "const"
    """
    p_flux_list = []
    comp_im_prob_fov_list = []
    comp_im_prob_sky_list = []
    for source in self.alldata:
      if source is not None:
        p_flux_list.append(source.best_fit_p_flux)
        if selected_sat == "const":
          comp_im_prob_fov_list.append(source.const_proba_compton_image_fov)
          comp_im_prob_sky_list.append(source.const_proba_compton_image_sky)
        else:
          comp_im_prob_fov_list.append(source.proba_compton_image_fov[selected_sat])
          comp_im_prob_sky_list.append(source.proba_compton_image_sky[selected_sat])
    distrib, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    distrib.suptitle(
      "Compton Image probability vs peak flux of detected long GRB - GRB in the whole sky (left) and only in the FoV (right)")
    ax1.scatter(p_flux_list, comp_im_prob_sky_list, s=2, label='Compton image probability over the whole sky')
    ax2.scatter(p_flux_list, comp_im_prob_fov_list, s=2, label='Compton image probability over the field of view')

    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability", xscale=x_scale, yscale=y_scale)
    ax1.legend()
    ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability", xscale=x_scale, yscale=y_scale)
    ax2.legend()
    plt.show()

  def mu100_distri(self, selected_sat="const", n_bins=30, x_scale='linear', y_scale="log"):
    """

    """
    if selected_sat == "const":
      title = "mu100 distribution of GRBs detected by the whole constellation"
    else:
      title = f"mu100 distribution of GRBs detected by the satellite {selected_sat}"

    mu_100_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.mu100 is not None:
                mu_100_list.append(sim.const_data.mu100)
            else:
              if sim[selected_sat] is not None:
                if sim[selected_sat].mu100 is not None:
                  mu_100_list.append(sim[selected_sat].mu100)
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    ax1.hist(mu_100_list, bins=n_bins, cumulative=0, histtype="step", weights=[self.weights] * len(mu_100_list))
    ax1.set(xlabel="mu100 (dimensionless)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale)
    plt.show()

  def mu100_vs_angle(self, selected_sat=0, x_scale='linear', y_scale="linear"):
    """

    """
    title = f"mu100 vs detection angle of GRBs detected by the satellite {selected_sat}"
    # Faire de façon à ce que tous les satellites détectés soient pris en compte aussi ~ const option
    mu_100_list = []
    angle_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if sim[selected_sat] is not None:
              if sim[selected_sat].mu100 is not None:
                mu_100_list.append(sim[selected_sat].mu100)
                angle_list.append(sim[selected_sat].dec_sat_frame)
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    ax1.scatter(mu_100_list, angle_list, s=2)
    ax1.set(xlabel="mu100 (dimensionless)", ylabel="Detection angle (°)", xlim=(0, 1), ylim=(115, 0), xscale=x_scale, yscale=y_scale)
    plt.show()

  def pa_distribution(self, selected_sat="const", n_bins=30, x_scale='linear', y_scale="log"):
    """

    """
    if selected_sat == "const":
      title = "Polarization angle distribution of GRBs detected by the whole constellation"
    else:
      title = f"Polarization angle distribution of GRBs detected by the satellite {selected_sat}"

    pa_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.pa is not None:
                pa_list.append(sim.const_data.pa)
            else:
              if sim[selected_sat] is not None:
                if sim[selected_sat].pa is not None:
                  pa_list.append(sim[selected_sat].pa)
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    ax1.hist(pa_list, bins=n_bins, cumulative=0, histtype="step", weights=[self.weights] * len(pa_list))
    ax1.set(xlabel="Polarization angle (°)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale)
    plt.show()

  def mdp_vs_fluence(self, selected_sat="const", mdp_threshold=1, x_scale='log', y_scale='linear'):
    """

    """
    if selected_sat == "const":
      title = "MDP as a function of fluence of GRBs detected by the whole constellation"
    else:
      title = f"MDP as a function of fluence of GRBs detected by the satellite {selected_sat}"

    if not self.cat_file == "None":
      mdp_list = []
      fluence_list = []
      mdp_count = 0
      no_detec_fluence = []
      for source_ite, source in enumerate(self.alldata):
        if source is not None:
          for sim in source:
            if sim is not None:
              if selected_sat == "const":
                if sim.const_data.mdp is not None:
                  if sim.const_data.mdp <= mdp_threshold:
                    mdp_list.append(sim.const_data.mdp * 100)
                    fluence_list.append(source.source_fluence)
                  else:
                    no_detec_fluence.append(source.source_fluence)
                mdp_count += 1
              else:
                if sim[selected_sat] is not None:
                  if sim[selected_sat].mdp is not None:
                    if sim[selected_sat].mdp <= mdp_threshold:
                      mdp_list.append(sim[selected_sat].mdp * 100)
                      fluence_list.append(source.source_fluence)
                    else:
                      no_detec_fluence.append(source.source_fluence)
                  mdp_count += 1

      distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
      distrib.suptitle(title)
      for ite_val, val in enumerate(np.unique(no_detec_fluence)):
        if ite_val == 0:
          ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
        else:
          ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
      ax1.scatter(fluence_list, mdp_list, s=3,
                  label=f'Detected GRB polarization \nRatio of detectable polarization : {len(mdp_list) / mdp_count}')
      ax1.set(xlabel="fluence (erg.cm-2)", ylabel="MDP (%)", xscale=x_scale, yscale=y_scale,
              xlim=(10 ** (int(np.log10(np.min(fluence_list))) - 1), 10 ** (int(np.log10(np.max(fluence_list))) + 1)))
      ax1.legend()
      plt.show()

  def mdp_vs_pflux(self, selected_sat="const", mdp_threshold=1, x_scale='log', y_scale='linear'):
    """

    """
    if selected_sat == "const":
      title = "MDP as a function of flux at peak of GRBs detected by the whole constellation"
    else:
      title = f"MDP as a function of flux at peak of GRBs detected by the satellite {selected_sat}"

    mdp_list = []
    flux_list = []
    mdp_count = 0
    no_detec_flux = []
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.mdp is not None:
                if sim.const_data.mdp <= mdp_threshold:
                  mdp_list.append(sim.const_data.mdp * 100)
                  flux_list.append(source.best_fit_p_flux)
                else:
                  no_detec_flux.append(source.best_fit_p_flux)
              mdp_count += 1
            else:
              if sim[selected_sat] is not None:
                if sim[selected_sat].mdp is not None:
                  if sim[selected_sat].mdp <= mdp_threshold:
                    mdp_list.append(sim[selected_sat].mdp * 100)
                    flux_list.append(source.best_fit_p_flux)
                  else:
                    no_detec_flux.append(source.best_fit_p_flux)
                mdp_count += 1

    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    for ite_val, val in enumerate(np.unique(no_detec_flux)):
      if ite_val == 0:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
      else:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
    ax1.scatter(flux_list, mdp_list, s=3,
                label=f'Detected GRB polarization \nRatio of detectable polarization : {len(mdp_list) / mdp_count}')
    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="MDP (%)", xscale=x_scale, yscale=y_scale,
            xlim=(10 ** (int(np.log10(np.min(flux_list))) - 1), 10 ** (int(np.log10(np.max(flux_list))) + 1)))
    ax1.legend()
    plt.show()

  def mdp_vs_detection_angle(self, selected_sat=0, mdp_threshold=1, x_scale='linear', y_scale='linear'):
    """

    """
    if selected_sat == "const":
      title = "MDP as a function of detection angle of GRBs detected by the whole constellation"
    else:
      title = f"MDP as a function of detection angle of GRBs detected by the satellite {selected_sat}"

    mdp_list = []
    angle_list = []
    mdp_count = 0
    no_detec_angle = []
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim in source:
          if sim is not None:
            if sim[selected_sat] is not None:
              if sim[selected_sat].mdp is not None:
                if sim[selected_sat].mdp <= mdp_threshold:
                  mdp_list.append(sim[selected_sat].mdp * 100)
                  angle_list.append(sim[selected_sat].dec_sat_frame)
                else:
                  no_detec_angle.append(sim[selected_sat].dec_sat_frame)
                mdp_count += 1

    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    for ite_val, val in enumerate(np.unique(no_detec_angle)):
      if ite_val == 0:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
      else:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
    ax1.scatter(angle_list, mdp_list, s=3,
                label=f'Detected GRB polarization \nRatio of detectable polarization : {len(mdp_list) / mdp_count}')
    ax1.set(xlabel="Angle (°)", ylabel="MDP (%)", xscale=x_scale, yscale=y_scale,
            xlim=(0, 180))
    ax1.legend()
    plt.show()

  def snr_vs_fluence(self, snr_type="compton", selected_sat="const", snr_threshold=5, x_scale='log', y_scale='log'):
    """

    """
    if selected_sat == "const":
      file_string = "the whole constellation"
    else:
      file_string = f"the satellite {selected_sat}"
    if snr_type == "compton":
      title = f"SNR of compton events as a function of fluence of GRBs detected by {file_string}"
    elif snr_type == "single":
      title = f"SNR of single events as a function of fluence of GRBs detected by {file_string}"
    else:
      print("Choose a correct type of snr : compton(default), single")
      return "snr_vs_fluence error"

    if not self.cat_file == "None":
      snr_list = []
      fluence_list = []
      snr_count = 0
      no_detec_fluence = []
      for source_ite, source in enumerate(self.alldata):
        if source is not None:
          for sim in source:
            if sim is not None:
              if selected_sat == "const":
                if snr_type == "compton":
                  if sim.const_data.snr_compton_t90 >= snr_threshold:
                    snr_list.append(sim.const_data.snr_compton_t90)
                    fluence_list.append(source.source_fluence)
                  else:
                    no_detec_fluence.append(source.source_fluence)
                elif snr_type == "single":
                  if sim.const_data.snr_single_t90 >= snr_threshold:
                    snr_list.append(sim.const_data.snr_single_t90)
                    fluence_list.append(source.source_fluence)
                  else:
                    no_detec_fluence.append(source.source_fluence)
                snr_count += 1
              else:
                if snr_type == "compton":
                  if sim[selected_sat] is not None:
                    if sim[selected_sat].snr_compton_t90 is not None:
                      if sim[selected_sat].snr_compton_t90 >= snr_threshold:
                        snr_list.append(sim[selected_sat].snr_compton_t90)
                        fluence_list.append(source.source_fluence)
                      else:
                        no_detec_fluence.append(source.source_fluence)
                      snr_count += 1
                elif snr_type == "single":
                  if sim[selected_sat] is not None:
                    if sim[selected_sat].snr_single_t90 is not None:
                      if sim[selected_sat].snr_single_t90 >= snr_threshold:
                        snr_list.append(sim[selected_sat].snr_single_t90)
                        fluence_list.append(source.source_fluence)
                      else:
                        no_detec_fluence.append(source.source_fluence)
                      snr_count += 1

      distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
      distrib.suptitle(title)
      for ite_val, val in enumerate(np.unique(no_detec_fluence)):
        if ite_val == 0:
          ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
        else:
          ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
      ax1.scatter(fluence_list, snr_list, s=3,
                  label=f'Detected GRB SNR \nRatio of detectable GRB : {len(snr_list) / snr_count}')
      ax1.set(xlabel="Fluence (erg.cm-2)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale,
              xlim=(10 ** (int(np.log10(np.min(fluence_list))) - 1), 10 ** (int(np.log10(np.max(fluence_list))) + 1)))
      ax1.legend()
      plt.show()

  def snr_vs_pflux(self, snr_type="compton", selected_sat="const", snr_threshold=5, x_scale='log', y_scale='log'):
    """

    """
    if selected_sat == "const":
      file_string = "the whole constellation"
    else:
      file_string = f"the satellite {selected_sat}"
    if snr_type == "compton":
      title = f"SNR of compton events as a function of peak flux of GRBs detected by {file_string}"
    elif snr_type == "single":
      title = f"SNR of single events as a function of peak flux of GRBs detected by {file_string}"
    else:
      print("Choose a correct type of snr : compton(default), single")
      return "snr_vs_fluence error"

    snr_list = []
    flux_list = []
    snr_count = 0
    no_detec_flux = []
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if snr_type == "compton":
                if sim.const_data.snr_compton_t90 >= snr_threshold:
                  snr_list.append(sim.const_data.snr_compton_t90)
                  flux_list.append(self.alldata[source_ite].best_fit_p_flux)
                else:
                  no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
              elif snr_type == "single":
                if sim.const_data.snr_single_t90 >= snr_threshold:
                  snr_list.append(sim.const_data.snr_single_t90)
                  flux_list.append(self.alldata[source_ite].best_fit_p_flux)
                else:
                  no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
              snr_count += 1
            else:
              if snr_type == "compton":
                if sim[selected_sat] is not None:
                  if sim[selected_sat].snr_compton_t90 is not None:
                    if sim[selected_sat].snr_compton_t90 >= snr_threshold:
                      snr_list.append(sim[selected_sat].snr_compton_t90)
                      flux_list.append(self.alldata[source_ite].best_fit_p_flux)
                    else:
                      no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
                    snr_count += 1
              elif snr_type == "single":
                if sim[selected_sat] is not None:
                  if sim[selected_sat].snr_single_t90 is not None:
                    if sim[selected_sat].snr_single_t90 >= snr_threshold:
                      snr_list.append(sim[selected_sat].snr_single_t90)
                      flux_list.append(self.alldata[source_ite].best_fit_p_flux)
                    else:
                      no_detec_flux.append(self.alldata[source_ite].best_fit_p_flux)
                    snr_count += 1

    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    for ite_val, val in enumerate(np.unique(no_detec_flux)):
      if ite_val == 0:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
      else:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
    ax1.scatter(flux_list, snr_list, s=3,
                label=f'Detected GRB SNR \nRatio of detectable GRB : {len(snr_list) / snr_count}')
    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale,
            xlim=(10 ** (int(np.log10(np.min(flux_list))) - 1), 10 ** (int(np.log10(np.max(flux_list))) + 1)))
    ax1.legend()
    plt.show()

  def snr_vs_detection_angle(self, snr_type="compton", selected_sat=0, snr_threshold=5, x_scale='linear', y_scale='log'):
    """

    """
    if selected_sat == "const":
      file_string = "the whole constellation"
    else:
      file_string = f"the satellite {selected_sat}"
    if snr_type == "compton":
      title = f"SNR of compton events as a function of detection angle of GRBs detected by {file_string}"
    elif snr_type == "single":
      title = f"SNR of single events as a function of detection angle of GRBs detected by {file_string}"
    else:
      print("Choose a correct type of snr : compton(default), single")
      return "snr_vs_fluence error"

    snr_list = []
    angle_list = []
    snr_count = 0
    no_detec_angle = []
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim in source:
          if sim is not None:
            if snr_type == "compton":
              if sim[selected_sat] is not None:
                if sim[selected_sat].snr_compton_t90 is not None:
                  if sim[selected_sat].snr_compton_t90 >= snr_threshold:
                    snr_list.append(sim[selected_sat].snr_compton_t90)
                    angle_list.append(sim[selected_sat].dec_sat_frame)
                  else:
                    no_detec_angle.append(sim[selected_sat].dec_sat_frame)
                  snr_count += 1
            elif snr_type == "single":
              if sim[selected_sat] is not None:
                if sim[selected_sat].snr_single_t90 is not None:
                  if sim[selected_sat].snr_single_t90 >= snr_threshold:
                    snr_list.append(sim[selected_sat].snr_single_t90)
                    angle_list.append(sim[selected_sat].dec_sat_frame)
                  else:
                    no_detec_angle.append(sim[selected_sat].dec_sat_frame)
                  snr_count += 1

    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle(title)
    for ite_val, val in enumerate(np.unique(no_detec_angle)):
      if ite_val == 0:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black', label="Markers for rejected GRB")
      else:
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
    ax1.scatter(angle_list, snr_list, s=3,
                label=f'Detected GRB SNR \nRatio of detectable GRB : {len(snr_list) / snr_count}')
    ax1.set(xlabel="Detection angle (°)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale,
            xlim=(0, 180))
    ax1.legend()
    plt.show()

  def mdp_vs_snr(self, selected_sat="const", snr_threshold=5, mdp_threshold=1, print_rejected=False,
                 x_scale='log', y_scale='linear'):
    """

    """
    if selected_sat == "const":
      title = "SNR as a function of MDP of GRBs detected by the whole constellation"
    else:
      title = f"SNR as a function of MDP of GRBs detected by the satellite {selected_sat}"

    mdp_list = []
    snr_list = []
    count = 0
    no_detec = [[], []]
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.snr_compton_t90 >= snr_threshold and sim.const_data.mdp <= mdp_threshold:
                mdp_list.append(sim.const_data.mdp * 100)
                snr_list.append(sim.const_data.snr_compton_t90)
              else:
                no_detec[0].append(sim.const_data.mdp * 100)
                no_detec[1].append(sim.const_data.snr_compton_t90)
              count += 1
            else:
              if sim[selected_sat] is not None:
                if sim[selected_sat].snr_compton_t90 is not None:
                  if sim[selected_sat].snr_compton_t90 >= snr_threshold and sim[selected_sat].mdp <= mdp_threshold:
                    mdp_list.append(sim[selected_sat].mdp * 100)
                    snr_list.append(sim[selected_sat].snr_compton_t90)
                  else:
                    no_detec[0].append(sim[selected_sat].mdp * 100)
                    no_detec[1].append(sim[selected_sat].snr_compton_t90)
                  count += 1

      distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
      distrib.suptitle(title)
      if print_rejected:
        ax1.scatter(no_detec[0], no_detec[1], label="Markers for rejected GRB")
      ax1.scatter(mdp_list, snr_list, s=3,
                  label=f'Detected GRB (Both with SNR and MDP) \nRatio of detectable GRB : {len(snr_list) / count}')
      ax1.set(xlabel="Fluence (erg.cm-2)", ylabel="SNR (dimensionless)", xscale=x_scale, yscale=y_scale)
      ax1.legend()
      plt.show()
