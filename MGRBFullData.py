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
mpl.use('Qt5Agg')
# plt.rcParams.update({'font.size': 20})


class GRBFullData:
  """
  Class containing the data for 1 GRB, for 1 sim, and 1 satellite
  """
  def __init__(self, data_list, sat_info, burst_time, sim_duration, num_sat, bkg_data, mu_data, ergcut, armcut,
               geometry, save_time, corr, polarigram_bins):
    """
    :param data_list: list of files to read (should be containing only 1)
    :param sat_info: orbital information about the satellite detecting the source
    :param burst_time: time at which the detection was made
    :param sim_duration: duration of the simulation
    :param num_sat: number of the satellite detecting the source
    :param bkg_data: list of background data to affect the correct count rates to this simulation
    :param mu_data: list of mu100 data to affect the correct mu100 and effective area to this simulation
    :param ergcut: energy cut to use
    :param armcut: ARM (Angular Resolution Measurement) cut to use
    :param geometry: geometry used for the simulation
    :param save_time: True if the time of interaction should be saved
    :param corr: True if the polarigrams should be corrected (useful when adding them together for the constellation)
    :param polarigram_bins: bins for the polarigram
    """
    ##############################################################
    #                   Attributes declaration                   #
    ##############################################################
    ##############################################################
    # Attributes for the sat
    self.compton_b_rate = 0
    self.single_b_rate = 0
    self.sat_dec_wf = None
    self.sat_ra_wf = None
    self.num_sat = num_sat
    ##############################################################
    # Attributes from the mu100 files
    self.mu100_ref = None
    self.mu100_err_ref = None
    self.s_eff_compton_ref = 0
    self.s_eff_single_ref = 0

    ##############################################################
    # Attributes filled with file reading (or to be used from this moment)
    self.grb_dec_sat_frame = None
    self.grb_ra_sat_frame = None
    self.compton_ener = []
    self.compton_second = []
    self.single_ener = []
    compton_firstpos = []
    compton_secpos = []
    single_pos = []
    if save_time:
      self.compton_time = []
      self.single_time = []
    self.pol = None
    self.polar_from_position = None
    # This polar angle is the one considered as compton scatter angle by mimrec
    self.polar_from_energy = None
    self.arm_pol = None
    self.azim_angle_corrected = False
    ##############################################################
    # Attributes filled after the reading
    # Set using extracted data
    self.s_eff_compton = 0
    self.s_eff_single = 0
    self.single = 0
    self.single_cr = 0
    self.compton = 0
    self.compton_cr = 0

    self.expected_pa = None
    self.bins = None
    self.mdp = None
    self.snr_compton = None
    self.snr_single = None
    self.snr_compton_t90 = None
    self.snr_single_t90 = None
    ##############################################################
    # Attributes that are used while making const
    self.n_sat_detect = 1
    # Attributes that are used while determining the deterctor where the interaction occured
    self.calor = 0
    self.dsssd = 0
    self.side = 0

    ##############################################################
    #                   Reading data from file                   #
    ##############################################################
    if len(data_list) == 0:  # Empty object for the constellation making
      self.n_sat_detect = 0
    else:
      dec_world_frame, ra_world_frame, source_name, num_sim, num_sat = fname2decra(data_list[0])
      self.expected_pa, self.grb_dec_sat_frame, self.grb_ra_sat_frame = grb_decrapol_worldf2satf(dec_world_frame, ra_world_frame, sat_info[0], sat_info[1])[:3]
      # Extracting the data from first file
      data_pol = readfile(data_list[0])
      for event in data_pol:
        reading = readevt(event, ergcut)
        if len(reading) == 5:
          # print("Reading\n")
          self.compton_second.append(reading[0])
          self.compton_ener.append(reading[1])
          if save_time:
            self.compton_time.append(reading[2])
          compton_firstpos.append(reading[3])
          compton_secpos.append(reading[4])
        elif len(reading) == 3:
          self.single_ener.append(reading[0])
          if save_time:
            self.single_time.append(reading[1])
          single_pos.append(reading[2])
      self.compton_ener = np.array(self.compton_ener)
      self.compton_second = np.array(self.compton_second)
      self.single_ener = np.array(self.single_ener)
      compton_firstpos = np.array(compton_firstpos)
      compton_secpos = np.array(compton_secpos)
      single_pos = np.array(single_pos)
      if save_time:
        self.compton_time = np.array(self.compton_time)
        self.single_time = np.array(self.single_time)

      ##############################################################
      #                     Filling the fields                     #
      ##############################################################
      # Calculating the polar angle with energy values and compton azim and polar scattering angles from the kinematics
      # polar and position angle stored in deg
      self.polar_from_energy = calculate_polar_angle(self.compton_second, self.compton_ener)
      self.pol, self.polar_from_position = angle(compton_secpos - compton_firstpos, self.grb_dec_sat_frame, self.grb_ra_sat_frame, source_name, num_sim, num_sat)

      # Calculating the arm and extracting the indexes of correct arm events (arm in deg)
      self.arm_pol = self.polar_from_position - self.polar_from_energy
      accepted_arm_pol = np.where(np.abs(self.arm_pol) <= armcut, True, False)
      # Restriction of the values according to arm cut
      self.compton_ener = self.compton_ener[accepted_arm_pol]
      self.compton_second = self.compton_second[accepted_arm_pol]
      compton_firstpos = compton_firstpos[accepted_arm_pol]
      compton_secpos = compton_secpos[accepted_arm_pol]
      if save_time:
        self.compton_time = self.compton_time[accepted_arm_pol]
      self.polar_from_energy = self.polar_from_energy[accepted_arm_pol]
      self.polar_from_position = self.polar_from_position[accepted_arm_pol]
      self.pol = self.pol[accepted_arm_pol]

      # Correcting the angle correction for azimuthal angle according to cosima's polarization definition
      # And setting the attribute stating if the correction is applied or not
      # Putting the correction before the filtering may cause some issues
      self.bins = set_bins(polarigram_bins, self.pol)
      if corr:
        self.corr()

      ##############################################################
      #        Filling the fields with bkg and mu100 files         #
      ##############################################################
      if sat_info is not None:
        # self.compton_b_rate = sat_info[-2]
        # self.single_b_rate = sat_info[-1]
        self.sat_dec_wf, self.sat_ra_wf, self.compton_b_rate, self.single_b_rate = affect_bkg(sat_info, burst_time, bkg_data)
      self.mu100_ref, self.mu100_err_ref, self.s_eff_compton_ref, self.s_eff_single_ref = closest_mufile(self.grb_dec_sat_frame, self.grb_ra_sat_frame, mu_data)
      # Putting the azimuthal scattering angle between the correct bins for creating histograms
      self.single = len(self.single_ener)
      self.single_cr = self.single / sim_duration
      self.compton = len(self.compton_ener)
      self.compton_cr = self.compton / sim_duration

      ##############################################################
      #     Finding the detector of interaction for each event     #
      ##############################################################
      self.compton_first_detector, self.compton_sec_detector, self.single_detector = find_detector(compton_firstpos, compton_secpos, single_pos, geometry)
      hits = np.array([])
      if self.compton_first_detector > 0:
        hits = np.concatenate((hits, self.compton_first_detector[:, 1]))
      if self.compton_sec_detector > 0:
        hits = np.concatenate((hits, self.compton_sec_detector[:, 1]))
      if self.single_detector > 0:
        hits = np.concatenate((hits, self.single_detector[:, 1]))
      self.calor = 0
      self.dsssd = 0
      self.side = 0
      for hit in hits:
        if hit == "Calor":
          self.calor += 1
        elif hit.startswith("SideDet"):
          self.side += 1
        elif hit.startswith("Layer"):
          self.dsssd += 1
        else:
          print("Error, unknown interaction volume")
      self.total_hits = self.calor + self.dsssd + self.side
      # TODO testing

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

  @staticmethod
  def get_keys():
    print("======================================================================")
    print("    Attributes")
    print(" Number of the satellite (or satellites if constellation):                .num_sat")
    print(" Number of triggers in the detectors                                      .triggers")
    print(" Number of triggers in the colorimeters                                   .calor")
    print(" Number of triggers in the dsssds                                         .dsssd")
    print(" Number of triggers in the side detectors                                 .side")
    print(" Number of single hits                                                    .single")
    print(" Number of compton events recorded                                        .compton")
    print(" Number of compton events recorded (from the field CE)                    .CE")
    print(" Compton events count rate of the source for the satellite/constellation  .cr")
    print(" List of the azimuthal compton angles for all Compton events )pol data)   .pol")
    print(" List of the azimuthal compton angles for all Compton events (unpol data) .unpol")
    print(" Number of sat detecting the source, != 1 only for constellations         .n_sat_detect")
    print(" mu100 for the satellite/constellation                                    .mu100")
    print(" Polarization angle obtained from the polarigram                          .pa")
    print(" Compton events count rate of the source from the fit                     .fit_cr")
    print(" Minimum detectable polarization calculated with Compton events           .mdp")
    print(" Signal to noise ratio calculated from a choosen field (ex :CE)           .snr")
    print(" Declination of the source in the world frame                             .dec_world_frame")
    print(" Right ascention of the source in the world frame                         .ra_world_frame")
    print(" Declination of the source in the satelitte(s) frame                      .dec_sat_frame")
    print(" Right ascention of the source in the satelitte(s) frame                  .ra_sat_frame")
    print(" Expected polarization angle for the satelittes(s) (set by sim)           .expected_pa")
    print(" Polarization angle error from the fit                                    .pa_err")
    print(" mu100 error from the fit                                                 .mu100_err")
    print(" Count rate error from the fit                                            .fit_cr_err")
    print(" Fit goodness                                                             .fit_goodness")

    print("======================================================================")
    print("    Methods")
    print("======================================================================")

  def analyze(self, source_duration, source_fluence):
    """
    Proceeds to the data analysis to get s_eff, mdp, snr
      mdp has physical significance between 0 and 1
    :param source_duration: duration of the source
    :param source_fluence: fluence of the source [photons/cm2]
    """
    if source_fluence is None:
      self.s_eff_compton = None
      self.s_eff_single = None
    else:
      self.s_eff_compton = self.compton_cr * source_duration / source_fluence
      self.s_eff_single = self.single_cr * source_duration / source_fluence

    self.mdp = calc_mdp(self.compton_cr * source_duration, self.compton_b_rate * source_duration, self.mu100_ref)
    # Calculation of SNR with 1sec of integration
    snr_compton_val = calc_snr(self.compton_cr, self.compton_b_rate)
    snr_single_t90_val = calc_snr(self.single_cr, self.single_b_rate)
    # Saving the snr for different integration times
    if snr_compton_val < 0:
      self.snr_compton_t90 = 0
      self.snr_compton = 0
    else:
      self.snr_compton_t90 = snr_compton_val * np.sqrt(source_duration)
      self.snr_compton = snr_compton_val
    if snr_single_t90_val < 0:
      self.snr_single_t90 = 0
      self.snr_single = 0
    else:
      self.snr_single_t90 = snr_single_t90_val * np.sqrt(source_duration)
      self.snr_single = snr_single_t90_val
