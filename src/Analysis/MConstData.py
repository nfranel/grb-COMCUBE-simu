# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  MConstData.py
# Class to contain GRB data for a given source, simulation, and constellation
# ================================================================

# Package imports
import numpy as np

# Developped modules imports
from src.General.funcmod import calc_mdp, calc_snr


class ConstData:
  """
  Class containing the data for 1 GRB, for 1 sim, and a given constellation
  """
  def __init__(self, num_offsat):
    """
    :param num_offsat: number of satellites not working in the constellation
    """
    ###################################################################################################################
    #  Attributes declaration    +    way they are treated with constellation
    ###################################################################################################################
    # Attributes for the sat
    self.bkg_index = None                       # Appened
    self.sat_mag_dec = None                     # Appened
    self.compton_b_rate = 0                     # Summed
    self.single_b_rate = 0                      # Summed
    self.num_sat = None                         # Appened
    self.num_offsat = num_offsat                # Not changed
    ###################################################################################################################
    # Attributes from the mu100 files
    self.mu100_ref = None                       # Weighted mean
    self.mu100_err_ref = None                   # Weighted mean
    self.s_eff_compton_ref = 0                  # Summed
    self.s_eff_single_ref = 0                   # Summed
    ###################################################################################################################
    # Attributes filled with file reading (or to be used from this moment)
    self.compton_time = []                      # 1D concatenation
    self.single_time = []                       # 1D concatenation
    ###################################################################################################################
    # Attributes filled after the reading
    # Set using extracted data
    self.s_eff_compton = 0                      # Summed
    self.s_eff_single = 0                       # Summed
    self.s_eff_compton_err = 0                  # propagated sum
    self.s_eff_single_err = 0                   # propagated sum
    self.single = 0                             # Summed
    self.single_cr = 0                          # Summed
    self.compton = 0                            # Summed
    self.compton_cr = 0                         # Summed

    self.mdp = None                             # Not changed
    self.mdp_err = None                         # Not changed
    self.hits_snrs = None                       # Not changed
    self.compton_snrs = None                    # Not changed
    self.single_snrs = None                     # Not changed
    self.hits_snrs_err = None                   # Not changed
    self.compton_snrs_err = None                # Not changed
    self.single_snrs_err = None                 # Not changed
    ###################################################################################################################
    # Attributes that are used while making const
    self.n_sat_detect = 0                       # Summed
    ###################################################################################################################
    self.const_beneficial_compton = True       # Appened
    self.const_beneficial_single = True        # Appened
    self.const_beneficial_trigger_4s = np.zeros(9, dtype=np.int16)  # List sum
    self.const_beneficial_trigger_3s = np.zeros(9, dtype=np.int16)  # List sum
    self.const_beneficial_trigger_2s = np.zeros(9, dtype=np.int16)  # List sum
    self.const_beneficial_trigger_1s = np.zeros(9, dtype=np.int16)  # List sum

  def analyze(self, source_duration, source_fluence):
    """
    Proceeds to the data analysis to get s_eff, modp, snr
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
      self.s_eff_compton_err = np.sqrt(self.compton_cr * source_duration / source_fluence**2 + (self.compton_cr * source_duration)**2 / source_fluence**3)
      self.s_eff_single_err = np.sqrt(self.single_cr * source_duration / source_fluence**2 + (self.single_cr * source_duration)**2 / source_fluence**3)

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
