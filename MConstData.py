# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import matplotlib.pyplot as plt
import matplotlib as mpl
# Developped modules imports
from funcmod import *

# Ploting adjustments
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')

# plt.rcParams.update({'font.size': 20})

class ConstData:
  """
  Class containing the data for 1 GRB, for 1 sim, and 1 satellite
  """
  def __init__(self, num_offsat):
    """
    :param data_list: list of files to read (should be containing only 1)
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
    # self.array_dtype = "float32"
    ###################################################################################################################
    # Attributes for the sat
    self.compton_b_rate = 0                     # Summed                  # Compton
    self.single_b_rate = 0                      # Summed                  # Single
    self.hit_b_rate = 0                         # Summed                  # Trigger quality selection
    # self.sat_dec_wf = None                      # Not changed             #
    # self.sat_ra_wf = None                       # Not changed             #
    # self.sat_alt = None                         # Not changed             #
    self.num_sat = None                         # Appened                 #
    self.num_offsat = num_offsat                   # Not changed                 #
    ###################################################################################################################
    # Attributes from the mu100 files
    self.mu100_ref = None                       # Weighted mean           # Compton
    self.mu100_err_ref = None                   # Weighted mean           # Compton
    self.s_eff_compton_ref = 0                  # Summed                  # Compton
    self.s_eff_single_ref = 0                   # Summed                  # Single

    ###################################################################################################################
    # Attributes filled with file reading (or to be used from this moment)
    # self.grb_dec_sat_frame = None               # Not changed             #
    # self.grb_ra_sat_frame = None                # Not changed             #
    # self.expected_pa = None                     # Not changed             #
    # self.compton_ener = []                      # 1D concatenation        # Compton
    # self.compton_second = []                    # 1D concatenation        # Compton
    # self.single_ener = []                       # 1D concatenation        # Single
    self.compton_time = []                      # 1D concatenation        # Compton
    self.single_time = []                       # 1D concatenation        # Single
    # self.hit_time = []                          # 1D concatenation        # Trigger quality selection
    # self.pol = None                             # 1D concatenation        # Compton
    # self.polar_from_position = None             # 1D concatenation        # Compton
    # This polar angle is the one considered as compton scatter angle by mimrec
    # self.polar_from_energy = None               # 1D concatenation        # Compton
    # self.arm_pol = None                         # 1D concatenation        # Compton
    # self.azim_angle_corrected = False           # Set to true             #
    ###################################################################################################################
    # interaction position attributes
    compton_firstpos = []
    compton_secpos = []
    single_pos = []
    # self.compton_first_detector = []            # 1D concatenation        # Compton
    # self.compton_sec_detector = []              # 1D concatenation        # Compton
    # self.single_detector = []                   # 1D concatenation        # Single
    ###################################################################################################################
    # Attributes filled after the reading
    # Set using extracted data
    self.s_eff_compton = 0                      # Summed                  # Compton
    self.s_eff_single = 0                       # Summed                  # Single
    self.single = 0                             # Summed                  # Single
    self.single_cr = 0                          # Summed                  # Single
    self.compton = 0                            # Summed                  # Compton
    self.compton_cr = 0                         # Summed                  # Compton

    # self.bins = None                            # All the same            #
    self.mdp = None                             # Not changed             #
    self.hits_snrs = None                       # Not changed             #
    self.compton_snrs = None                    # Not changed             #
    self.single_snrs = None                     # Not changed             #
    ###################################################################################################################
    # Attributes that are used while making const
    self.n_sat_detect = 0                       # Summed                  #
    # Attributes that are used while determining the deterctor where the interaction occured
    self.calor = 0                              # Summed                  # Trigger quality selection ?
    self.dsssd = 0                              # Summed                  # Trigger quality selection ?
    self.side = 0                               # Summed                  # Trigger quality selection ?
    self.total_hits = 0
    ###################################################################################################################
    self.const_beneficial_compton = True       # Appened                 #
    self.const_beneficial_single = True        # Appened                 #
    self.const_beneficial_trigger_4s = np.zeros(9, dtype=np.int16)  # List sum                  #
    self.const_beneficial_trigger_3s = np.zeros(9, dtype=np.int16)  # List sum                  #
    self.const_beneficial_trigger_2s = np.zeros(9, dtype=np.int16)  # List sum                  #
    self.const_beneficial_trigger_1s = np.zeros(9, dtype=np.int16)  # List sum                  #

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

    #################################################################################################################
    # Calculation of effective area
    #################################################################################################################
    self.calculates_snrs(source_duration)

    #################################################################################################################
    # Calculation of mdp
    #################################################################################################################
    self.mdp = calc_mdp(self.compton_cr * source_duration, self.compton_b_rate * source_duration, self.mu100_ref)

  def calculates_snrs(self, source_duration):
    """
    Calculates the snr for different integration time
    :param source_duration: duration of the source
    """
    integration_times = [0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096, source_duration]

    self.hits_snrs = []
    self.compton_snrs = []
    self.single_snrs = []

    for int_time in integration_times:
      bins = np.arange(0, source_duration + int_time, int_time)

      hit_hist = np.histogram(np.concatenate((self.compton_time, self.single_time)), bins=bins)[0]
      hit_argmax = np.argmax(hit_hist)
      hit_max_hist = hit_hist[hit_argmax]
      self.hits_snrs.append(calc_snr(hit_max_hist, self.hit_b_rate * int_time))

      compton_hist = np.histogram(self.compton_time, bins=bins)[0]
      com_argmax = np.argmax(compton_hist)
      com_max_hist = compton_hist[com_argmax]
      self.compton_snrs.append(calc_snr(com_max_hist, self.compton_b_rate * int_time))

      single_hist = np.histogram(self.single_time, bins=bins)[0]
      sin_argmax = np.argmax(single_hist)
      sin_max_hist = single_hist[sin_argmax]
      self.single_snrs.append(calc_snr(sin_max_hist, self.single_b_rate * int_time))
