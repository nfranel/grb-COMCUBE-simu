# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Developped modules imports
from funcmod import *

# Ploting adjustments
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')

# plt.rcParams.update({'font.size': 20})


class GRBFullData:
  """
  Class containing the data for 1 GRB, for 1 sim, and 1 satellite
  """

  def __init__(self, datafile, sim_duration, source_duration, source_fluence, corr, polarigram_bins):
    """
    :param datafile: file to read
    :param sat_info: orbital information about the satellite detecting the source
    :param burst_time: time at which the detection was made
    :param sim_duration: duration of the simulation
    :param bkg_data: list of background data to affect the correct count rates to this simulation
    :param mu_data: list of mu100 data to affect the correct mu100 and effective area to this simulation
    :param ergcut: energy cut to use
    :param armcut: ARM (Angular Resolution Measurement) cut to use
    :param geometry: geometry used for the simulation
    :param corr: True if the polarigrams should be corrected (useful when adding them together for the constellation)
    :param polarigram_bins: bins for the polarigram
    """
    ###################################################################################################################
    #  Attributes declaration    +    way they are treated with constellation
    ###################################################################################################################
    self.array_dtype = np.float32
    ###################################################################################################################
    # Attributes for the sat
    self.bkg_index = None                  # Appened
    self.compton_b_rate = 0                # Summed                  # Compton
    self.single_b_rate = 0                 # Summed                  # Single
    self.sat_dec_wf = None                 # Not changed             #
    self.sat_ra_wf = None                  # Not changed             #
    self.sat_alt = None                    # Not changed             #
    self.num_offsat = None                 # Not changed             #
    self.num_sat = None                    # Appened                 #
    ###################################################################################################################
    # Attributes from the mu100 files
    self.mu100_ref = None                  # Weighted mean           # Compton
    self.mu100_err_ref = None              # Weighted mean           # Compton
    self.s_eff_compton_ref = 0             # Summed                  # Compton
    self.s_eff_single_ref = 0              # Summed                  # Single
    ###################################################################################################################
    # Attributes filled with file reading (or to be used from this moment)
    self.grb_dec_sat_frame = None          # Not changed             #
    self.grb_ra_sat_frame = None           # Not changed             #
    self.expected_pa = None                # Not changed             #
    self.compton_ener = []                 # 1D concatenation        # Compton
    self.compton_second = []               # 1D concatenation        # Compton
    self.single_ener = []                  # 1D concatenation        # Single
    self.compton_time = []                 # 1D concatenation        # Compton
    self.single_time = []                  # 1D concatenation        # Single
    self.pol = None                        # 1D concatenation        # Compton
    # self.pol_err = None   !update makeconst                      # 1D concatenation        # Compton
    self.polar_from_position = None        # 1D concatenation        # Compton
    # This polar angle is the one considered as compton scatter angle by mimrec
    self.polar_from_energy = None          # 1D concatenation        # Compton
    # self.polar_from_energy_err = None    !update makeconst           # 1D concatenation        # Compton
    self.arm_pol = None                    # 1D concatenation        # Compton
    self.azim_angle_corrected = False      # Set to true             #
    ###################################################################################################################
    # interaction position attributes
    self.compton_first_detector = []       # 1D concatenation        # Compton
    self.compton_sec_detector = []         # 1D concatenation        # Compton
    self.single_detector = []              # 1D concatenation        # Single
    ###################################################################################################################
    # Attributes filled after the reading
    # Set using extracted data
    self.s_eff_compton = 0                 # Summed                  # Compton
    self.s_eff_single = 0                  # Summed                  # Single
    self.s_eff_compton_err = 0             # propagated sum
    self.s_eff_single_err = 0              # propagated sum
    self.single = 0                        # Summed                  # Single
    self.single_cr = 0                     # Summed                  # Single
    self.compton = 0                       # Summed                  # Compton
    self.compton_cr = 0                    # Summed                  # Compton

    self.bins = None                       # All the same            #
    self.mdp = None                        # Not changed             #
    self.mdp_err = None                    # Not changed             #
    self.hits_snrs = None                  # Not changed             #
    self.compton_snrs = None               # Not changed             #
    self.single_snrs = None                # Not changed             #
    self.hits_snrs_err = None              # Not changed             #
    self.compton_snrs_err = None           # Not changed             #
    self.single_snrs_err = None            # Not changed             #
    ###################################################################################################################
    # Attributes that are used while making const
    self.n_sat_detect = 1                  # Summed                  #
    ###################################################################################################################
    self.const_beneficial_compton = True   # Appened                 #
    self.const_beneficial_single = True    # Appened                 #
    self.const_beneficial_trigger_4s = np.zeros(9, dtype=np.int16)  # List sum                  #
    self.const_beneficial_trigger_3s = np.zeros(9, dtype=np.int16)  # List sum                  #
    self.const_beneficial_trigger_2s = np.zeros(9, dtype=np.int16)  # List sum                  #
    self.const_beneficial_trigger_1s = np.zeros(9, dtype=np.int16)  # List sum                  #

    ###################################################################################################################
    #                   Reading data from file
    ###################################################################################################################
    if datafile is None:  # Empty list object for the constellation making
      self.n_sat_detect = 0
    elif type(datafile) is str:
      #################################################################################################################
      #                     Filling the fields by reading the extracted sim files
      #################################################################################################################
      try:
        self.read_saved_grb(datafile)
      except:
        print(traceback.format_exc())
        print(f"Error happened with file : {datafile}")
      #################################################################################################################
      #        Counting events
      #################################################################################################################
      self.single = len(self.single_ener)
      self.single_cr = self.single / sim_duration
      self.compton = len(self.compton_ener)
      self.compton_cr = self.compton / sim_duration

      #################################################################################################################
      #        Setting bins and correcting the polarigram
      #################################################################################################################
      # Correcting the angle correction for azimuthal angle according to cosima's polarization definition
      # And setting the attribute stating if the correction is applied or not
      # Putting the correction before the filtering may cause some issues
      self.bins = set_bins(polarigram_bins, self.pol)
      if corr:
        self.corr()

      #################################################################################################################
      #        Conducting other calculations
      #################################################################################################################
      # TODO testing
      self.analyze(source_duration, source_fluence)

      self.set_beneficial_compton()
      self.set_beneficial_single()
      self.set_beneficial_trigger()
    else:
      raise TypeError("Impossible to create the data container : the data must be None or a string")

  def read_saved_grb(self, filename):
    with open(filename, "r") as f:
      # lines = f.read().split("\n")[2:]
      # Nothing do with the first 2 lines
      next(f)
      next(f)

      # Specific to satellite
      self.bkg_index = int(next(f))
      self.sat_dec_wf = float(next(f))
      self.sat_ra_wf = float(next(f))
      self.sat_alt = float(next(f))
      self.num_sat = int(next(f))
      self.compton_b_rate = float(next(f))
      self.single_b_rate = float(next(f))
      # Information from mu files
      self.mu100_ref = float(next(f))
      self.mu100_err_ref = float(next(f))
      self.s_eff_compton_ref = float(next(f))
      self.s_eff_single_ref = float(next(f))
      # GRB position and polarisation
      self.grb_dec_sat_frame = float(next(f))
      self.grb_ra_sat_frame = float(next(f))
      self.expected_pa = float(next(f))
      # Value arrays
      self.compton_ener = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.compton_second = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.single_ener = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.compton_time = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.single_time = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.pol = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.polar_from_position = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.polar_from_energy = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.arm_pol = np.fromstring(next(f), sep='|', dtype=self.array_dtype)
      self.compton_first_detector = np.fromstring(next(f), sep='|', dtype=np.int8)
      self.compton_sec_detector = np.fromstring(next(f), sep='|', dtype=np.int8)
      self.single_detector = np.fromstring(next(f), sep='|', dtype=np.int8)

    if len(self.compton_first_detector) == 1 and self.compton_first_detector[0] == "":
      self.compton_first_detector = np.array([])
    if len(self.compton_sec_detector) == 1 and self.compton_sec_detector[0] == "":
      self.compton_sec_detector = np.array([])
    if len(self.single_detector) == 1 and self.single_detector[0] == "":
      self.single_detector = np.array([])
    if len(self.single_detector) != len(self.single_time):
      print("============ERROR=============")

  def cor(self):
    """
    Calculates the angle to correct for the source sky position and cosima's "RelativeY" polarization definition
    :returns: float, angle in deg
    Warning : That's actually minus the correction angle (so that the correction uses a + instead of a - ...)
    """
    theta, phi = np.deg2rad(self.grb_dec_sat_frame), np.deg2rad(self.grb_ra_sat_frame)
    return np.rad2deg(np.arctan(np.cos(theta) * np.tan(phi))) + self.expected_pa

  def behave(self, width=360):
    """
    Make angles be between the beginning of the first bin and the beginning of the first bin plus the width parameter
    Calculi are made in-place
    :param width: float, width of the polarigram in deg, default=360, SHOULD BE 360
    """
    self.pol = self.pol % width + self.bins[0]

  def corr(self):
    """
    Corrects the angles from the source sky position and cosima's "RelativeY" polarization definition
    """
    if self.azim_angle_corrected:
      print(" Impossible to correct the azimuthal compton scattering angles, the correction has already been made")
    else:
      cor = self.cor()
      self.pol += cor
      self.behave()
      self.azim_angle_corrected = True

  def anticorr(self):
    """
    Undo the corr operation
    """
    if self.azim_angle_corrected:
      cor = self.cor()
      self.pol -= cor
      self.behave()
      self.azim_angle_corrected = False
    else:
      print(" Impossible to undo the correction of the azimuthal compton scattering angles : no correction were made")

  def analyze(self, source_duration, source_fluence):
    """
    Proceeds to the data analysis to get s_eff, mdp, snr
      mdp has physical significance between 0 and 1
    :param source_duration: duration of the source
    :param source_fluence: fluence of the source [photons/cm2]
    """
    #################################################################################################################
    # Calculation of effective area
    #################################################################################################################
    if source_fluence is None:
      self.s_eff_compton = None
      self.s_eff_single = None
    else:
      self.s_eff_compton = self.compton_cr * source_duration / source_fluence
      self.s_eff_single = self.single_cr * source_duration / source_fluence
      # Error of source fluence taken as only a poissonian error as we consider the spectrum as a fixed information
      self.s_eff_compton_err = np.sqrt(self.compton_cr * source_duration / source_fluence ** 2 + (self.compton_cr * source_duration) ** 2 / source_fluence ** 3)
      self.s_eff_single_err = np.sqrt(self.single_cr * source_duration / source_fluence ** 2 + (self.single_cr * source_duration) ** 2 / source_fluence ** 3)

    #################################################################################################################
    # Calculation of effective area
    #################################################################################################################
    self.calculates_snrs(source_duration)

    #################################################################################################################
    # Calculation of mdp
    #################################################################################################################
    self.mdp, self.mdp_err = calc_mdp(self.compton_cr * source_duration, self.compton_b_rate * source_duration, self.mu100_ref, mu100_err=self.mu100_err_ref)

  def calculates_snrs(self, source_duration):
    """
    Calculates the snr for different integration time
    :param source_duration: duration of the source
    """
    integration_times = [0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096, source_duration]

    self.hits_snrs = []
    self.compton_snrs = []
    self.single_snrs = []
    self.hits_snrs_err = []
    self.compton_snrs_err = []
    self.single_snrs_err = []

    for int_time in integration_times:
      bins = np.arange(0, source_duration + int_time, int_time)

      hit_hist = np.histogram(np.concatenate((self.compton_time, self.compton_time, self.single_time)), bins=bins)[0]
      hit_argmax = np.argmax(hit_hist)
      hit_max_hist = hit_hist[hit_argmax]
      snr_ret1 = calc_snr(hit_max_hist, (2 * self.compton_b_rate + self.single_b_rate) * int_time)
      self.hits_snrs.append(snr_ret1[0])
      self.hits_snrs_err.append(snr_ret1[1])

      compton_hist = np.histogram(self.compton_time, bins=bins)[0]
      com_argmax = np.argmax(compton_hist)
      com_max_hist = compton_hist[com_argmax]
      snr_ret2 = calc_snr(com_max_hist, self.compton_b_rate * int_time)
      self.compton_snrs.append(snr_ret2[0])
      self.compton_snrs_err.append(snr_ret2[1])

      single_hist = np.histogram(self.single_time, bins=bins)[0]
      sin_argmax = np.argmax(single_hist)
      sin_max_hist = single_hist[sin_argmax]
      snr_ret3 = calc_snr(sin_max_hist, self.single_b_rate * int_time)
      self.single_snrs.append(snr_ret3[0])
      self.single_snrs_err.append(snr_ret3[1])

  def hits_snrs_over_lc(self, source_duration, nsat=3):
    if nsat == 2:
      thresh_list_nsat = [5.7, 5.3, 5, 4.8, 4.6, 4.4, 4.3, 4.2, 4.1]
    elif nsat == 3:
      thresh_list_nsat = [4.7, 4.3, 4.2, 4, 3.8, 3.7, 3.6, 3.5, 3.4]
    elif nsat == 4:
      thresh_list_nsat = [4.1, 3.9, 3.7, 3.6, 3.5, 3.3, 3.3, 3.2, 3.1]
    else:
      raise ValueError("Uncorrect number of sat : only 2, 3, and 4 sat constellation are considered")
    integration_times = [0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096]
    hits_snrs_lc = []
    for ite_int, int_time in enumerate(integration_times):
      bins = np.arange(0, source_duration + int_time, int_time)
      hit_hist = np.histogram(np.concatenate((self.compton_time, self.compton_time, self.single_time)), bins=bins)[0]
      hits_snrs_lc.append(np.where(calc_snr(hit_hist, (2 * self.compton_b_rate + self.single_b_rate) * int_time)[0] >= thresh_list_nsat[ite_int], 1, 0))
    return hits_snrs_lc

  def set_beneficial_compton(self, threshold=2.6):
    """
    Sets const_beneficial_compton to True is the value for a satellite is worth considering
    :param threshold: the mdp threshold required to consider a satellite is worth
    """
    if self.mdp < threshold:
      self.const_beneficial_compton = True
    else:
      self.const_beneficial_compton = False

  def set_beneficial_single(self):
    """
    Sets const_beneficial_compton to True is the value for a satellite is worth considering
    """
    self.const_beneficial_single = True

  def set_beneficial_trigger(self):
    """
    Sets const_beneficial_compton to True is the value for a satellite is worth considering
    """
    # For 4 sats
    # thresh_list has the sigma threshold for 16, 32, 64, 128, 256, 512, 1024, 2048 and 4096s
    thresh_list_4s = [4.1, 3.9, 3.7, 3.6, 3.5, 3.3, 3.3, 3.2, 3.1]
    for ite_ts in range(len(self.const_beneficial_trigger_4s)):
      if self.hits_snrs[ite_ts] >= thresh_list_4s[ite_ts]:
        self.const_beneficial_trigger_4s[ite_ts] = 1
      else:
        self.const_beneficial_trigger_4s[ite_ts] = 0

    # For 3 sats
    thresh_list_3s = [4.7, 4.3, 4.2, 4, 3.8, 3.7, 3.6, 3.5, 3.4]
    for ite_ts in range(len(self.const_beneficial_trigger_3s)):
      if self.hits_snrs[ite_ts] >= thresh_list_3s[ite_ts]:
        self.const_beneficial_trigger_3s[ite_ts] = 1
      else:
        self.const_beneficial_trigger_3s[ite_ts] = 0

    # For 2 sats
    thresh_list_2s = [5.7, 5.3, 5, 4.8, 4.6, 4.4, 4.3, 4.2, 4.1]
    for ite_ts in range(len(self.const_beneficial_trigger_2s)):
      if self.hits_snrs[ite_ts] >= thresh_list_2s[ite_ts]:
        self.const_beneficial_trigger_2s[ite_ts] = 1
      else:
        self.const_beneficial_trigger_2s[ite_ts] = 0

    # For 1 sats
    thresh_list_1s = [8.2, 7.5, 6.9, 6.6, 6.3, 6, 5.8, 5.6, 5.5]
    for ite_ts in range(len(self.const_beneficial_trigger_1s)):
      if self.hits_snrs[ite_ts] >= thresh_list_1s[ite_ts]:
        self.const_beneficial_trigger_1s[ite_ts] = 1
      else:
        self.const_beneficial_trigger_1s[ite_ts] = 0

  def detector_statistics(self, bkg_cont, bkg_duration, source_duration, source_name, show=False):
    bkg_stats = (bkg_cont.det_stat_compton + bkg_cont.det_stat_single).reshape(4, 5) / bkg_duration

    hit_times = np.concatenate((self.compton_time, self.compton_time, self.single_time))
    det_list = np.concatenate((self.compton_first_detector, self.compton_sec_detector, self.single_detector))
    bin_edges = np.arange(0, source_duration + 1, 1)
    bin_index = np.digitize(hit_times, bin_edges) - 1
    hit_hist = np.histogram(hit_times, bins=bin_edges)[0]

    # print("=================== Digitize verif ===================")
    # for iteval in range(len(bin_index)):
    #   print(f"{hit_times[iteval]} - {bin_index[iteval]} - {det_list[iteval]}")
    # print("=================== Digitize verif ===================")

    binned_dets = [[] for ite in range(len(bin_edges) - 1)]
    for ite_ev, idx in enumerate(bin_index):
      if bin_edges[idx] <= hit_times[ite_ev] < bin_edges[idx + 1]:
        binned_dets[idx].append(det_list[ite_ev])

    # print("=================== Binning verif ===================")
    # for itebin, bin in enumerate(binned_dets):
    #   print(f"Bin {itebin}")
    #   print(f"hit_hist and binned_dets same size : {hit_hist[itebin] == len(binned_dets[itebin])}")
    # print("=================== Binning verif ===================")

    shaped_det_lc = np.transpose(np.array([det_counter(np.array(binned_det)) for binned_det in binned_dets]), (1, 2, 0))
    # dets_lc = np.array([det_counter(np.array(binned_det)) for binned_det in binned_dets])
    # shaped_det_lc = np.transpose(dets_lc, (1, 2, 0))
    # print("=================== Transposition verif ===================")
    # test_cont = np.zeros((4, 5, len(dets_lc)))
    # for ite_lc in range(len(dets_lc)):
    #   # Bin de la lc
    #   for ite_quad in range(len(dets_lc[ite_lc])):
    #     # Dans le quad
    #     for ite_det in range(len(dets_lc[ite_lc][ite_quad])):
    #       test_cont[ite_quad][ite_det][ite_lc] = dets_lc[ite_lc][ite_quad][ite_det]

    # print(shaped_det_lc)
    # print(test_cont)
    # print(f"transposition size and looped size : {shaped_det_lc.shape} - {test_cont.shape}")
    # if shaped_det_lc.shape == test_cont.shape:
    #   print(f"shaped_det_lc == test_cont : {np.all(shaped_det_lc == test_cont)}")
    # print("=================== Transposition verif ===================")

    if show:
      fig, axes = plt.subplots(4, 5)
      fig.suptitle(f"{source_name}")
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
          ax.stairs(bkg_stats[itequad][itedet] + shaped_det_lc[itequad][itedet], bin_edges, fill=True, edgecolor="blue")
          ax.axhline(bkg_stats[itequad][itedet], color="green")

    max_dets_stats = np.max(shaped_det_lc, axis=2)

    return bkg_stats + max_dets_stats
