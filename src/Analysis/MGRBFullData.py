# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  MGRBFullData.py
# Class to contain GRB data for a given source, simulation, and satellite
# ================================================================


# Package imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import traceback

# Developped modules imports
from src.General.funcmod import set_bins, calc_mdp, calc_snr, det_counter


class GRBFullData:
  """
  Class containing the data for 1 GRB, for 1 sim, and 1 satellite
  """

  def __init__(self, datafile, sim_duration, source_duration, source_fluence, bkgdata, mudata, options):
    """
    :param datafile: Simulation file to read
    :param sim_duration: Duration of the simulation
    :param source_duration: Duration of the source emission
    :param source_fluence: Source fluence
    :param bkgdata: background container to affect the correct count rates to this simulation
    :param mudata: mu100/Seff container to affect the correct mu100 and effective area to this simulation
    :param options: list containing [erg_cut, armcut, geometry, init_correction, polarigram_bins]
    options gives information about the following options, required for data analysis
        ergcut: Energy cut to use
        armcut: ARM (Angular Resolution Measurement) cut to use
        geometry: geometry of the mass model used for the simulation
        init_correction: True if the polarigrams should be corrected (useful when adding them together for the constellation)
        polarigram_bins: Bins for the polarigram
    """
    ergcut, armcut, corr, polarigram_bins = options[0], options[1], options[3], options[4]
    ###################################################################################################################
    #  Attributes declaration    +    way they are combined to obtain constellation level data
    ###################################################################################################################
    self.array_dtype = np.float32
    ###################################################################################################################
    # Attributes for the sat
    self.bkg_index = None                  # Appened
    self.sat_mag_dec = None                # Appened
    self.compton_b_rate = 0                # Summed
    self.single_b_rate = 0                 # Summed
    self.sat_dec_wf = None                 # Not changed
    self.sat_ra_wf = None                  # Not changed
    self.sat_alt = None                    # Not changed
    self.num_offsat = None                 # Not changed
    self.num_sat = None                    # Appened
    ###################################################################################################################
    # Attributes from the mu100 files
    self.mu_index = None                   # Appened
    self.mu100_ref = None                  # Weighted mean
    self.mu100_err_ref = None              # Weighted mean
    self.s_eff_compton_ref = 0             # Summed
    self.s_eff_single_ref = 0              # Summed
    ###################################################################################################################
    # Attributes filled with file reading (or to be used from this moment)
    self.grb_dec_sat_frame = None          # Not changed
    self.grb_ra_sat_frame = None           # Not changed
    self.expected_pa = None                # Not changed
    self.df_compton = None
    self.df_single = None
    ###################################################################################################################
    # Correction applied
    self.azim_angle_corrected = False      # Set to true
    ###################################################################################################################
    # Attributes filled after the reading
    # Set using extracted data
    self.s_eff_compton = 0                 # Summed
    self.s_eff_single = 0                  # Summed
    self.s_eff_compton_err = 0             # propagated sum
    self.s_eff_single_err = 0              # propagated sum
    self.single = 0                        # Summed
    self.single_cr = 0                     # Summed
    self.compton = 0                       # Summed
    self.compton_cr = 0                    # Summed

    self.bins = None                       # All the same
    self.mdp = None                        # Not changed
    self.mdp_err = None                    # Not changed
    self.hits_snrs = None                  # Not changed
    self.compton_snrs = None               # Not changed
    self.single_snrs = None                # Not changed
    self.hits_snrs_err = None              # Not changed
    self.compton_snrs_err = None           # Not changed
    self.single_snrs_err = None            # Not changed
    ###################################################################################################################
    # Attributes that are used while making const
    self.n_sat_detect = 1                  # Summed
    ###################################################################################################################
    self.const_beneficial_compton = True   # Appened
    self.const_beneficial_single = True    # Appened
    self.const_beneficial_trigger_4s = np.zeros(9, dtype=np.int16)  # List sum
    self.const_beneficial_trigger_3s = np.zeros(9, dtype=np.int16)  # List sum
    self.const_beneficial_trigger_2s = np.zeros(9, dtype=np.int16)  # List sum
    self.const_beneficial_trigger_1s = np.zeros(9, dtype=np.int16)  # List sum

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
        self.read_saved_grb(datafile, bkgdata, mudata, ergcut, armcut)
      except:
        print(traceback.format_exc())
        print(f"Error happened with file : {datafile}")
      #################################################################################################################
      #        Counting events
      #################################################################################################################
      self.single = len(self.df_single)
      self.single_cr = self.single / sim_duration
      self.compton = len(self.df_compton)
      self.compton_cr = self.compton / sim_duration

      #################################################################################################################
      #        Setting bins and correcting the polarigram
      #################################################################################################################
      # Correcting the angle correction for azimuthal angle according to cosima's polarization definition
      # And setting the attribute stating if the correction is applied or not
      # Putting the correction before the filtering may cause some issues
      self.bins = set_bins(polarigram_bins, self.df_compton.pol.values)
      if corr:
        self.corr()

      #################################################################################################################
      #        Conducting other calculations
      #################################################################################################################
      self.analyze(source_duration, source_fluence)

      self.set_beneficial_compton()
      self.set_beneficial_single()
      self.set_beneficial_trigger()
    else:
      raise TypeError("Impossible to create the data container : the data must be None or a string")

  def read_saved_grb(self, filename, bkgdata, mudata, ergcut=None, armcut=None):
    """
    Method to read the HDFS files containing the condensed GRB simulation data
    :param filename: Name of the condensed data file
    :param bkgdata: Background data container
    :param mudata: mu100/Seff data container
    :param ergcut: Energy cut applied during the analysis
    :param armcut: ARM (Angular Resolution Measurement) cut applied during the analysis
    """
    with pd.HDFStore(filename, mode="r") as f:
      self.df_compton = f["compton"]
      self.df_single = f["single"]

      if ergcut is not None:
        self.df_compton = self.df_compton[(self.df_compton.compton_ener >= ergcut[0]) & (self.df_compton.compton_ener <= ergcut[1])]
        self.df_single = self.df_single[(self.df_single.single_ener >= ergcut[0]) & (self.df_single.single_ener <= ergcut[1])]

      if armcut is not None:
        self.df_compton = self.df_compton[self.df_compton.arm_pol <= armcut]

      # Specific to satellite
      self.bkg_index = f.get_storer("compton").attrs.b_idx
      self.sat_mag_dec = f.get_storer("compton").attrs.sat_mag_dec
      self.sat_dec_wf = f.get_storer("compton").attrs.sat_dec_wf
      self.sat_ra_wf = f.get_storer("compton").attrs.sat_ra_wf
      self.sat_alt = f.get_storer("compton").attrs.sat_alt
      self.num_sat = f.get_storer("compton").attrs.num_sat
      # Information from mu files
      self.mu_index = f.get_storer("compton").attrs.mu_index
      # GRB position and polarisation
      self.grb_dec_sat_frame = f.get_storer("compton").attrs.grb_dec_sat_frame
      self.grb_ra_sat_frame = f.get_storer("compton").attrs.grb_ra_sat_frame
      self.expected_pa = f.get_storer("compton").attrs.expected_pa

      # print("comp des valeurs bkg: ", bkgdata.bkgdf.bkg_dec.values[self.bkg_index], self.sat_mag_dec)
      self.compton_b_rate = bkgdata.bkgdf.compton_cr.values[self.bkg_index]
      self.single_b_rate = bkgdata.bkgdf.single_cr.values[self.bkg_index]

      # print("comp des valeurs mu : ", mudata.mudf.bkg_dec.values[self.bkg_index], self.sat_mag_dec)
      self.mu100_ref = mudata.mudf.mu100.values[self.mu_index]
      self.mu100_err_ref = mudata.mudf.mu100_err.values[self.mu_index]
      self.s_eff_compton_ref = mudata.mudf.seff_compton.values[self.mu_index]
      self.s_eff_single_ref = mudata.mudf.seff_single.values[self.mu_index]

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
    Makes angles be between the beginning of the first bin and the beginning of the first bin plus the width parameter
    Calculi are made in-place
    :param width: float, width of the polarigram in deg, default=360, SHOULD BE 360
    """
    self.df_compton.pol = self.df_compton.pol % width + self.bins[0]

  def corr(self):
    """
    Corrects the angles from the source sky position and cosima's "RelativeY" polarization definition
    """
    if self.azim_angle_corrected:
      print(" Impossible to correct the azimuthal compton scattering angles, the correction has already been made")
    else:
      cor = self.cor()
      self.df_compton.pol += cor
      self.behave()
      self.azim_angle_corrected = True

  def anticorr(self):
    """
    Undo the corr operation
    """
    if self.azim_angle_corrected:
      cor = self.cor()
      self.df_compton.pol -= cor
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
    Calculates the maximum SNR for different integration time
      compton_snrs corresponds to the SNR of compton events only
      single_snrs corresponds to the SNR of single events only
      hits_snrs corresponds to the SNR of both types of events
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

      hit_hist = np.histogram(np.concatenate((self.df_compton.compton_time.values, self.df_compton.compton_time.values, self.df_single.single_time.values)), bins=bins)[0]
      hit_argmax = np.argmax(hit_hist)
      hit_max_hist = hit_hist[hit_argmax]
      snr_ret1 = calc_snr(hit_max_hist, (2 * self.compton_b_rate + self.single_b_rate) * int_time)
      self.hits_snrs.append(snr_ret1[0])
      self.hits_snrs_err.append(snr_ret1[1])

      compton_hist = np.histogram(self.df_compton.compton_time.values, bins=bins)[0]
      com_argmax = np.argmax(compton_hist)
      com_max_hist = compton_hist[com_argmax]
      snr_ret2 = calc_snr(com_max_hist, self.compton_b_rate * int_time)
      self.compton_snrs.append(snr_ret2[0])
      self.compton_snrs_err.append(snr_ret2[1])

      single_hist = np.histogram(self.df_single.single_time.values, bins=bins)[0]
      sin_argmax = np.argmax(single_hist)
      sin_max_hist = single_hist[sin_argmax]
      snr_ret3 = calc_snr(sin_max_hist, self.single_b_rate * int_time)
      self.single_snrs.append(snr_ret3[0])
      self.single_snrs_err.append(snr_ret3[1])

  def hits_snrs_over_lc(self, source_duration, nsat=3):
    """
    Calculates the SNR over the light curve for various integration times to check if the detection threshold is exceeded
    This method aims at performing detection estimation
    :param source_duration: Duration of the source
    :param nsat: Number of satellites considered for the simultaneous detection
    """
    # Setting the threshold for detection according to the number of satellite
    if nsat == 1:
      thresh_list_nsat = [8.2, 7.5, 6.9, 6.6, 6.3, 6, 5.8, 5.6, 5.5]
    elif nsat == 2:
      thresh_list_nsat = [5.7, 5.3, 5, 4.8, 4.6, 4.4, 4.3, 4.2, 4.1]
    elif nsat == 3:
      thresh_list_nsat = [4.7, 4.3, 4.2, 4, 3.8, 3.7, 3.6, 3.5, 3.4]
    elif nsat == 4:
      thresh_list_nsat = [4.1, 3.9, 3.7, 3.6, 3.5, 3.3, 3.3, 3.2, 3.1]
    else:
      raise ValueError("Uncorrect number of sat : only 2, 3, and 4 sat constellation are considered")

    # Calculation of the snr over the light curve for various integration times to check if the detection threshold is exceeded
    integration_times = [0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096]
    hits_snrs_lc = []
    for ite_int, int_time in enumerate(integration_times):
      bins = np.arange(0, source_duration + int_time, int_time)
      hit_hist = np.histogram(np.concatenate((self.df_compton.compton_time.values, self.df_compton.compton_time.values, self.df_single.single_time.values)), bins=bins)[0]
      hits_snrs_lc.append(np.where(calc_snr(hit_hist, (2 * self.compton_b_rate + self.single_b_rate) * int_time)[0] >= thresh_list_nsat[ite_int], 1, 0))
    return hits_snrs_lc

  def set_beneficial_compton(self, threshold=2.6):
    """
    Sets const_beneficial_compton to True is the value for a satellite is worth considering
    Objective : Filter data from satellites where adding the data would not be beneficial
    :param threshold: the mdp threshold required to consider a satellite is worth
    """
    if self.mdp < threshold:
      self.const_beneficial_compton = True
    else:
      self.const_beneficial_compton = False

  def set_beneficial_single(self):
    """
    Sets const_beneficial_compton to True is the value for a satellite is worth considering
    Objective : Filter data from satellites where adding the data would not be beneficial
    Set to True as no filtering is made on single events
    """
    self.const_beneficial_single = True

  def set_beneficial_trigger(self):
    """
    Sets const_beneficial_compton to True is the value for a satellite is worth considering
    Objective : Create arrays indicating if the detection threshold is exceeded for different number of satellites and different integration times
    This is used for a first fast and coarse detection method
    """
    # thresh_list_ns has the sigma threshold for 16, 32, 64, 128, 256, 512, 1024, 2048 and 4096s for n satellites
    # For 4 sats
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
    """
    Method used to estimate the count rate of each detector
    :param bkg_cont: Background data container
    :param bkg_duration: Background duration
    :param source_duration: Source duration
    :param source_name: Source name
    :param show: True if the plot of the light curve for each detector should be shown
    """
    bkg_stats = (bkg_cont.com_det_idx + bkg_cont.sin_det_idx).reshape(4, 5) / bkg_duration

    hit_times = np.concatenate((self.df_compton.compton_time.values, self.df_compton.compton_time.values, self.df_single.single_time.values))
    det_list = np.concatenate((self.df_compton.compton_first_detector.values, self.df_compton.compton_sec_detector.values, self.df_single.single_detector.values))
    bin_edges = np.arange(0, source_duration + 1, 1)
    bin_index = np.digitize(hit_times, bin_edges) - 1

    binned_dets = [[] for _ in range(len(bin_edges) - 1)]
    for ite_ev, idx in enumerate(bin_index):
      if bin_edges[idx] <= hit_times[ite_ev] < bin_edges[idx + 1]:
        binned_dets[idx].append(det_list[ite_ev])

    shaped_det_lc = np.transpose(np.array([det_counter(np.array(binned_det)) for binned_det in binned_dets]), (1, 2, 0))

    if show:
      mpl.use('Qt5Agg')
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
