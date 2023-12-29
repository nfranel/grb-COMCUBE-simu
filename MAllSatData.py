# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import subprocess
# Developped modules imports
from funcmod import *
from MGRBFullData import GRBFullData


class AllSatData(list):
  """
  Class containing all the data for 1 simulation of 1 GRB (or other source) for a full set of trafiles
  """

  def __init__(self, source_prefix, num_sim, sat_info, sim_duration, bkg_data, mu_data, options):
    temp_list = []
    # Attributes relative to the simulations without any analysis
    self.n_sat_receiving = 0
    self.n_sat = len(sat_info)
    self.dec_world_frame = None
    self.ra_world_frame = None
    self.grb_burst_time = None
    self.loading_count = 0
    for num_sat in range(self.n_sat):
      flist = subprocess.getoutput("ls {}_sat{}_{:04d}_*.inc1.id1.extracted.tra".format(source_prefix, num_sat, num_sim)).split("\n")
      if not flist[0].startswith("ls: cannot access") and self.dec_world_frame is None:
        self.dec_world_frame, self.ra_world_frame, self.grb_burst_time = fname2decratime(flist[0])[:3]
      if len(flist) == 1:
        if flist[0].startswith("ls: cannot access"):
          temp_list.append(None)
        else:
          temp_list.append(GRBFullData(flist, sat_info[num_sat], self.grb_burst_time, sim_duration, num_sat, bkg_data, mu_data, *options))
          self.n_sat_receiving += 1
          self.loading_count += 1
      else:
        print(f'WARNING : Unusual number of file : {flist}')
    list.__init__(self, temp_list)
    # Attribute meaningful after the creation of the constellation
    self.const_data = None

  @staticmethod
  def get_keys():
    print("======================================================================")
    print("    Attributes")
    print(" Number of satellites detecting the source :           .n_sat_det")
    print(" Number of satellites in the simulation :              .n_sat")
    print(" Declination of the source (world frame) :             .dec_world_frame")
    print(" Right ascention of the source (world frame) :         .ra_world_frame")
    print(" ===== Attribute that needs to be handled + 2 cases (full FoV or full sky)")
    print(" Extracted data but for a given set of satellites      .const_data")
    print("======================================================================")
    print("    Methods")
    print("======================================================================")

  def analyze(self, source_message, source_duration, source_fluence, fit_bounds, const_analysis):
    """
    Proceed to the analysis of polarigrams for all satellites and constellation (unless specified)
    """
    # First step of the analyze, to obtain the polarigrams, and values for polarization and snr
    for sat_ite, sat in enumerate(self):
      if sat is not None:
        sat.analyze(f"{source_message}sat {sat_ite}", source_duration, source_fluence, fit_bounds)
    if self.const_data is not None and const_analysis:
      self.const_data.analyze(f"{source_message}const", source_duration, source_fluence, fit_bounds)
    else:
      print("Constellation not set : please use make_const method if you want to analyze the constellation's results")

  def make_const(self, options, const=None):
    if const is None:
      const = np.array(range(self.n_sat))
    considered_sat = const[np.where(np.array(self) == None, False, True)]
    self.const_data = GRBFullData([], None, None, None, None, None, None, *options)

    for item in self.const_data.__dict__.keys():
      if item not in ["sat_dec_wf", "sat_ra_wf", "num_sat", "grb_dec_sat_frame", "grb_ra_sat_frame", "expected_pa", "fits", "mu100", "pa", "fit_compton_cr", "pa_err",
                      "mu100_err", "fit_compton_cr_err", "fit_goodness", "mdp", "snr_compton", "snr_single", "snr_compton_t90", "snr_single_t90"]:
        ###############################################################################################################
        # Values supposed to be the same for all sat and all sims so it doesn't change and is set using 1 sat
        # Except for polarigram error, its size is the same but the values depend on the fits
        if item in ["bins", "polarigram_error", "azim_angle_corrected"]:
          setattr(self.const_data, item, getattr(self[considered_sat[0]], item))
          print("debug", getattr(self[considered_sat[0]], item), getattr(self.const_data, item))
        ###############################################################################################################
        # Value set to True
        elif item in ["azim_angle_corrected"]:
          setattr(self.const_data, item, True)
        ###############################################################################################################
        # Values summed
        elif item in ["compton_b_rate", "single_b_rate", "s_eff_compton_ref", "s_eff_single_ref", "s_eff_compton", "s_eff_single", "single", "single_cr",
                      "compton", "compton_cr", "n_sat_detect", "calor", "dsssd", "side"]:
          temp_val = 0
          for num_sat in considered_sat:
            temp_val += getattr(self[num_sat], item)
          setattr(self.const_data, item, temp_val)
        ###############################################################################################################
        # Values stored in a 1D array that have to be concanated (except unpol that needs another verification)
        elif item in ["compton_ener", "compton_second", "single_ener", "compton_time", "single_time", "pol",
                      "polar_from_position", "polar_from_energy", "arm_pol"]:
          temp_array = np.array([])
          for num_sat in considered_sat:
            temp_array = np.concatenate((temp_array, getattr(self[num_sat], item)))
          setattr(self.const_data, item, temp_array)
        ###############################################################################################################
        # Values stored in a 2D array that have to be initiated and treated so that no error occur
        elif item in ["compton_firstpos", "compton_secpos", "single_pos"]:
          if len(considered_sat) == 1:
            if len(getattr(self[considered_sat[0]], item)) == 0:
              setattr(self.const_data, item, np.array([]))
            else:
              setattr(self.const_data, item, getattr(self[considered_sat[0]], item))
          else:
            temp_array = np.array([[0, 0, 0]])
            for ite_num_sat in range(len(considered_sat)):
              if len(getattr(self[considered_sat[ite_num_sat]], item)) == 0:
                temp_array = np.array([[0, 0, 0]])
              else:
                temp_array = np.concatenate((temp_array, getattr(self[considered_sat[ite_num_sat]], item)))
            setattr(self.const_data, item, temp_array[1:])
        ###############################################################################################################
        # unpol key
        elif item == "unpol":
          if getattr(self[considered_sat[0]], item) is not None:
            temp_array = np.array([])
            for num_sat in considered_sat:
              temp_array = np.concatenate((temp_array, getattr(self[num_sat], item)))
            setattr(self.const_data, item, temp_array)
        ###############################################################################################################
        # mu100_ref and mu100_err_ref key
        elif item in ["mu100_ref", "mu100_err_ref"]:
          temp_num = 0
          temp_denom = 0
          for num_sat in considered_sat:
            temp_num += getattr(self[num_sat], item) * self[num_sat].compton
            temp_denom += self[num_sat].compton
          if temp_denom != 0:
            setattr(self.const_data, item, temp_num / temp_denom)


  def verif_const(self, message="", const=None):
    """
    Method to check that the constellation has been done properly
    """
    if const is None:
      const = np.array(range(self.n_sat))
    considered_sat = const[np.where(np.array(self) == None, False, True)]

    for item in self.const_data.__dict__.keys():
      #      Non modified items. They stay as :
      # ["num_sat", "dec_sat_frame", "ra_sat_frame", "expected_pa"]
      # [None, None, None, None]
      #      Or, until analyze() method has been applied, remain the same :
      # ["fits", "mu100", "pa", "fit_compton_cr", "pa_err", "mu100_err", "fit_compton_cr_err", "fit_goodness", "mdp",
      # "snr_compton", "snr_single"]
      # [None, None, None, None, None, None, None, None, None, None, None]
      if item not in ["sat_dec_wf", "sat_ra_wf", "num_sat", "grb_dec_sat_frame", "grb_ra_sat_frame", "expected_pa", "fits", "mu100", "pa", "fit_compton_cr", "pa_err",
                      "mu100_err", "fit_compton_cr_err", "fit_goodness", "mdp", "snr_compton", "snr_single", "snr_compton_t90", "snr_single_t90"]:
        ###############################################################################################################
        # Non modified items set to None
        if item in ["num_sat", "dec_sat_frame", "ra_sat_frame", "expected_pa"]:
          if getattr(self.const_data, item) is not None:
            print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
        ###############################################################################################################
        # Value set to True
        elif item in ["azim_angle_corrected"]:
          setattr(self.const_data, item, True)
        ###############################################################################################################
        # Values supposed to be the same for all sat and all sims so it doesn't change and is set using 1 sat
        # Except for polarigram error, only is size doesn't change, hence this verification
        elif item in ["bins", "polarigram_error"]:
          verification_bool = False
          for num_sat in considered_sat:
            if len(getattr(self[num_sat], item)) != len(getattr(self.const_data, item)):
              verification_bool = True
          if verification_bool:
            print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
        ###############################################################################################################
        # Values supposed to be the same
        elif item in ["azim_angle_corrected"]:
          verification_bool = False
          for num_sat in considered_sat:
            if getattr(self[num_sat], item) != getattr(self.const_data, item):
              print(getattr(self[num_sat], item), getattr(self.const_data, item))
              verification_bool = True
          if verification_bool:
            print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
        ###############################################################################################################
        # Values summed
        elif item in ["compton_b_rate", "single_b_rate", "s_eff_compton_ref", "s_eff_single_ref", "s_eff_compton", "s_eff_single", "single", "single_cr",
                      "compton", "compton_cr", "n_sat_detect", "calor", "dsssd", "side"]:
          temp_val = 0
          for num_sat in considered_sat:
            temp_val += getattr(self[num_sat], item)
          if temp_val != getattr(self.const_data, item):
            print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
            print(f"  The compared values from the satelitte and the constellation are : {temp_val} & {getattr(self.const_data, item)}")
        ###############################################################################################################
        # Values stored in a 1D array that have to be concanated (except unpol that needs another verification)
        elif item in ["compton_ener", "compton_second", "single_ener", "compton_time", "single_time", "pol",
                      "polar_from_position", "polar_from_energy", "arm_pol"]:
          temp_val = 0
          for num_sat in considered_sat:
            temp_val += len(getattr(self[num_sat], item))
          if temp_val != len(getattr(self.const_data, item)):
            print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
        ###############################################################################################################
        # Values stored in a 2D array that have to be initiated and treated so that no error occur
        elif item in ["compton_firstpos", "compton_secpos", "single_pos"]:
          temp_val = 0
          for num_sat in considered_sat:
            temp_val += len(getattr(self[num_sat], item))
          if temp_val != len(getattr(self.const_data, item)):
            print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
        ###############################################################################################################
        # unpol key
        elif item == "unpol":
          if getattr(self[considered_sat[0]], item) is not None:
            temp_val = 0
            for num_sat in considered_sat:
              temp_val += len(getattr(self[num_sat], item))
            if temp_val != len(getattr(self.const_data, item)):
              print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
          else:
            if getattr(self.const_data, item) is not None:
              print(f"Anomaly detected in the setting of the item {item} by make_const {message}")
