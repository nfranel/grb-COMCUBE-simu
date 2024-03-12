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
  def __init__(self, source_prefix, num_sim, sat_info, sim_duration, bkg_data, mu_data, info_source, options):
    """
    :param source_prefix: prefix used for simulations + source name
    :param num_sim: number of the simulation
    :param sat_info: orbital information on the satellites
    :param sim_duration: duration of the simulation
    :param bkg_data: list containing the background data
    :param mu_data: list containing the mu100 data
    :param options: options for the analysis, defined in AllSourceData
    """
    temp_list = []
    # Attributes relative to the simulations without any analysis
    self.n_sat_receiving = 0
    self.n_sat = len(sat_info)
    self.dec_world_frame = None
    self.ra_world_frame = None
    self.grb_burst_time = None
    self.loading_count = 0
    for num_sat in range(self.n_sat):
      flist = subprocess.getoutput(f"ls {source_prefix}_sat{num_sat}_{num_sim:04d}_*.inc1.id1.extracted.tra").split("\n")
      if not flist[0].startswith("ls: cannot access") and self.dec_world_frame is None:
        self.dec_world_frame, self.ra_world_frame, self.grb_burst_time = fname2decratime(flist[0])[:3]
      if len(flist) == 1:
        if flist[0].startswith("ls: cannot access"):
          temp_list.append(None)
        else:
          temp_list.append(GRBFullData(flist, sat_info[num_sat], self.grb_burst_time, sim_duration, num_sat, bkg_data, mu_data, *info_source, *options))
          self.n_sat_receiving += 1
          self.loading_count += 1
      else:
        print(f'WARNING : Unusual number of file : {flist}')
    list.__init__(self, temp_list)
    # Attribute meaningful after the creation of the constellation
    self.const_data = None

  def analyze(self, source_duration, source_fluence, sats_analysis):
    """
    Proceed to the analysis for all satellites and constellation (unless specified)
    :param source_duration: duration of the source
    :param source_fluence: fluence of the source
    :param sats_analysis: whether the satellites should be analyzed
    """
    if sats_analysis:
      for sat_ite, sat in enumerate(self):
        if sat is not None:
          sat.analyze(source_duration, source_fluence)
    if self.const_data is not None:
      for constellation in self.const_data:
        constellation.analyze(source_duration, source_fluence)
    else:
      print("Constellation not set : please use make_const method if you want to analyze the constellation's results")

  def make_const(self, source_duration, source_fluence, off_sats, options, const=None):
    """
    Creates a constellation of several satellites by putting together the results
    :param source_duration: duration of the source
    :param source_fluence: fluence of the source
    :param off_sats: indexes of the down satellites
    :param options: options for the analysis, defined in AllSourceData
    :param const: array with the number of the satellite to put in the constellation
      If None all satellites are considered
    """
    if const is None:
      const = np.array(range(self.n_sat))
    ###################################################################################################################
    # Required for usual constellation
    ###################################################################################################################
    in_sight_sat = np.where(np.array(self) == None, False, True)
    sat_const = const[in_sight_sat]
    const_0off_data = GRBFullData([], None, None, None, None, None, None, source_duration, source_fluence, *options)
    ###################################################################################################################
    # Constellation with 1 off satellite
    ###################################################################################################################
    off_sat = off_sats[0]
    in_sight_sat_1off = in_sight_sat
    in_sight_sat_1off[off_sat] = False
    sat_const_1off = const[in_sight_sat_1off]
    const_1off_data = GRBFullData([], None, None, None, None, None, None, source_duration, source_fluence, *options)
    ###################################################################################################################
    # Constellation with 2 off satellites
    ###################################################################################################################
    off_sat1 = off_sats[1]
    off_sat2 = off_sats[2]
    in_sight_sat_2off = in_sight_sat
    in_sight_sat_2off[off_sat1] = False
    in_sight_sat_2off[off_sat2] = False
    sat_const_2off = const[in_sight_sat_2off]
    const_2off_data = GRBFullData([], None, None, None, None, None, None, source_duration, source_fluence, *options)
    ###################################################################################################################
    # Making of the constellations
    ###################################################################################################################
    list_considered_sat = [sat_const, sat_const_1off, sat_const_2off]
    self.const_data = [const_0off_data, const_1off_data, const_2off_data]
    for ite_const, considered_sats in enumerate(list_considered_sat):
      for item in self.const_data[ite_const].__dict__.keys():
        ###############################################################################################################
        # Not changed
        ###############################################################################################################
        # The fieldselected here stay as they are with their basic initialisation (most of the time None)
        # Fields to be used soon : "fits", "pa", "fit_compton_cr", "pa_err", "fit_compton_cr_err", "fit_goodness",
        if item not in ["sat_dec_wf", "sat_ra_wf", "grb_dec_sat_frame", "grb_ra_sat_frame", "expected_pa",
                        "mdp", "hits_snrs", "compton_snrs", "single_snrs"]:
          #############################################################################################################
          # Filtering the satellites for some items
          #############################################################################################################
          if item in ["compton_b_rate", "mu100_ref", "mu100_err_ref", "s_eff_compton_ref", "compton_ener",
                      "compton_second", "compton_time", "pol", "polar_from_position", "polar_from_energy", "arm_pol",
                      "s_eff_compton", "compton", "compton_cr"]:
            selected_sats = []
            for index_sat in considered_sats:
              if self[index_sat].const_beneficial_compton:
                selected_sats.append(index_sat)
            selected_sats = np.array(selected_sats)
          elif item in ["single_b_rate", "s_eff_single_ref", "single_ener", "single_time", "s_eff_single", "single",
                        "single_cr"]:
            selected_sats = []
            for index_sat in considered_sats:
              if self[index_sat].const_beneficial_single:
                selected_sats.append(index_sat)
            selected_sats = np.array(selected_sats)
          elif item in ["hit_b_rate", "hit_time", "calor", "dsssd", "side"]:
            selected_sats = []
            for index_sat in considered_sats:
              if self[index_sat].const_beneficial_trigger:
                selected_sats.append(index_sat)
            selected_sats = np.array(selected_sats)
          else:
            selected_sats = considered_sats
          #############################################################################################################
          # All the same
          #############################################################################################################
          # Values supposed to be the same for all sat and all sims so it doesn't change and is set using 1 sat
          # Field to be used soon : "polarigram_error"
          if item in ["bins"]:
            setattr(self.const_data[ite_const], item, getattr(self[selected_sats[0]], item))
          #############################################################################################################
          # Set to true
          #############################################################################################################
          elif item in ["azim_angle_corrected"]:
            setattr(self.const_data[ite_const], item, True)
          #############################################################################################################
          # Summed
          #############################################################################################################
          # Values summed
          elif item in ["compton_b_rate", "single_b_rate", "hit_b_rate", "s_eff_compton_ref", "s_eff_single_ref",
                        "s_eff_compton", "s_eff_single", "single", "single_cr", "compton", "compton_cr", "n_sat_detect",
                        "calor", "dsssd", "side"]:
            temp_val = 0
            for num_sat in selected_sats:
              temp_val += getattr(self[num_sat], item)
            setattr(self.const_data[ite_const], item, temp_val)
          #############################################################################################################
          # 1D concatenation
          #############################################################################################################
          # Values stored in a 1D array that have to be concanated (except unpol that needs another verification)
          elif item in ["compton_ener", "compton_second", "single_ener", "hit_time", "compton_time", "single_time",
                        "pol", "polar_from_position", "polar_from_energy", "arm_pol"]:
            temp_array = np.array([])
            for num_sat in selected_sats:
              temp_array = np.concatenate((temp_array, getattr(self[num_sat], item)))
            setattr(self.const_data[ite_const], item, temp_array)
          #############################################################################################################
          # 2D concatenation     - Not used anymore, but still there just in case
          #############################################################################################################
          # Values stored in a 2D array that have to be initiated and treated so that no error occur
          # elif item in ["compton_firstpos", "compton_secpos", "single_pos"]:
          #   if len(selected_sats) == 1:
          #     if len(getattr(self[selected_sats[0]], item)) == 0:
          #       setattr(constellations[ite_const], item, np.array([]))
          #     else:
          #       setattr(constellations[ite_const], item, getattr(self[selected_sats[0]], item))
          #   else:
          #     temp_array = np.array([[0, 0, 0]])
          #     for ite_num_sat in range(len(selected_sats)):
          #       if len(getattr(self[selected_sats[ite_num_sat]], item)) != 0:
          #         temp_array = np.concatenate((temp_array, getattr(self[selected_sats[ite_num_sat]], item)))
          #     setattr(constellations[ite_const], item, temp_array[1:])
          #############################################################################################################
          # Unpol          - Not used anymore, but still there just in case
          #############################################################################################################
          # elif item == "unpol":
          #   if getattr(self[selected_sats[0]], item) is not None:
          #     temp_array = np.array([])
          #     for num_sat in selected_sats:
          #       temp_array = np.concatenate((temp_array, getattr(self[num_sat], item)))
          #     setattr(constellations[ite_const], item, temp_array)
          #############################################################################################################
          # Weighted mean
          #############################################################################################################
          # mu100_ref and mu100_err_ref key
          elif item in ["mu100_ref", "mu100_err_ref"]:
            temp_num = 0
            temp_denom = 0
            for num_sat in selected_sats:
              temp_num += getattr(self[num_sat], item) * self[num_sat].compton
              temp_denom += self[num_sat].compton
            if temp_denom != 0:
              setattr(self.const_data[ite_const], item, temp_num / temp_denom)
            else:
              setattr(self.const_data[ite_const], item, 0)
          #############################################################################################################
          # Appened
          #############################################################################################################
          elif item in ["num_sat", "const_beneficial_compton", "const_beneficial_single", "const_beneficial_trigger"]:
            temp_list = []
            for num_sat in selected_sats:
              temp_list.append(getattr(self[num_sat], item))
            setattr(self.const_data[ite_const], item, np.array(temp_list))
          else:
            raise AttributeError("Item not found")
