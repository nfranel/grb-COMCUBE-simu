# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  MAllSatData.py
# Class to contain GRB satellite and constellation data for a given source and simulation
# ================================================================

# Package imports
import numpy as np
import pandas as pd

# Developped modules imports
from src.Analysis.MGRBFullData import GRBFullData
from src.Analysis.MConstData import ConstData


class AllSatData(list):
  """
  Class containing all the data for 1 simulation of 1 GRB (or other source) for a full set of trafiles
  """
  def __init__(self, all_sat_data, sat_info, sim_duration, info_source, bkgdata, mudata, options):
    """
    :param all_sat_data: 1D list containing simulation filenames
    :param sat_info: Orbital information on the satellites
    :param sim_duration: Duration of the simulation
    :param info_source: list containing [source_duration, source_fluence]
        source_duration: Duration of the source
        source_fluence: Fluence of the source
    :param bkgdata: Background data container
    :param mudata: mu100/Seff data container
    :param options: list containing [erg_cut, armcut, geometry, init_correction, polarigram_bins]
    options gives information about the following options, required for data analysis
        ergcut: Energy cut to use
        armcut: ARM (Angular Resolution Measurement) cut to use
        geometry: geometry of the mass model used for the simulation
        init_correction: True if the polarigrams should be corrected (useful when adding them together for the constellation)
        polarigram_bins: Bins for the polarigram
    """
    # Attributes relative to the simulations without any analysis
    self.n_sat_receiving = 0
    self.n_sat = len(sat_info)
    self.dec_world_frame = None
    self.ra_world_frame = None
    self.grb_burst_time = None

    # Setting grb world frame dec, ra and burst time
    self.read_grb_siminfo(all_sat_data)

    list.__init__(self, [GRBFullData(grb_ext_file, sim_duration, *info_source, bkgdata, mudata, options) if grb_ext_file is not None else None for grb_ext_file in all_sat_data])
    self.n_sat_receiving = len(self) - self.count(None)

    # Initializing the const_data key, that will be containing the constellation data container
    self.const_data = None

  def read_grb_siminfo(self, filelist):
    """
    Method to read the DEC, RA and time of the simulated GRB in the world frame
    Simulations for the different satellites all share the same DEC, RA and burst time so the information is taken from the first file read
    """
    for filename in filelist:
      if filename is not None:
        with pd.HDFStore(filename, mode="r") as f:
          self.dec_world_frame = f.get_storer("compton").attrs.dec_world_frame
          self.ra_world_frame = f.get_storer("compton").attrs.ra_world_frame
          self.grb_burst_time = f.get_storer("compton").attrs.burst_time
        return

  def analyze(self, source_duration, source_fluence, sats_analysis=True):
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
        if constellation is not None:
          constellation.analyze(source_duration, source_fluence)
    else:
      print("Constellation not set : please use make_const method if you want to analyze the constellation's results")

  def make_const(self, num_of_down_per_const, off_sats, const=None, dysfunction_enabled=True):
    """
    Creates a one or more constellations of several satellites by putting together the results
        Used when the data container for the constellation is the same as the one for the individual satellites (not condensed)
    :param num_of_down_per_const: List containing the number of satellite not working in the constellations (taken randomly)
        Gives the number of constellations wanted and their number of off satellites
        Example : if only one constellation is wanted with 2 satellites not working  num_of_down_per_const = [2]
        Example : if 3 constellation are wanted with 0, 2 and 5 satellites not working  num_of_down_per_const = [0, 2, 5]
    :param off_sats: list containing the indexes of satellites not working
    :param const: array with the indexes of the satellites to put in the constellation
        If None all satellites are considered
    :param dysfunction_enabled: True if dysfunction of satellites is taken into account (the value of the ints in num_of_down_per_const)
    """
    if const is None:
      const = np.array(range(self.n_sat))
    ###################################################################################################################
    # Required for usual constellation
    ###################################################################################################################
    in_sight_sat = np.where(np.array(self) == None, False, True)
    ###################################################################################################################
    # Preparing which satellite data will be combined
    ###################################################################################################################
    list_considered_sat = []
    self.const_data = []
    if dysfunction_enabled:
      number_const = len(num_of_down_per_const)
    else:
      number_const = 1
    for const_ite in range(number_const):
      self.const_data.append(GRBFullData(None, None, None, None, None, None, [None, None, None, None, None]))
      if off_sats[const_ite] is None:
        self.const_data[const_ite].num_offsat = 0
      elif type(off_sats[const_ite]) is list:
        self.const_data[const_ite].num_offsat = len(off_sats[const_ite])
      else:
        raise TypeError("Error while making the constellations : the off_sats variable must be None or a list of index")
      in_sight_temp = in_sight_sat
      if off_sats[const_ite] is not None:
        for index in off_sats[const_ite]:
          in_sight_temp[index] = False
      sat_considered_temp = const[in_sight_temp]
      list_considered_sat.append(sat_considered_temp)
    ###################################################################################################################
    # Combining satellite data
    ###################################################################################################################
    for ite_const, considered_sats in enumerate(list_considered_sat):
      if len(considered_sats) == 0:
        self.const_data[ite_const] = None
      else:
        for item in self.const_data[ite_const].__dict__.keys():
          ###############################################################################################################
          # Not changed
          ###############################################################################################################
          # The fieldselected here stay as they are with their basic initialisation (most of the time None)
          # Fields to be used soon : "fits", "pa", "fit_compton_cr", "pa_err", "fit_compton_cr_err", "fit_goodness",
          if item not in ["sat_dec_wf", "sat_ra_wf", "sat_alt", "grb_dec_sat_frame", "grb_ra_sat_frame", "expected_pa",
                          "mdp", "mdp_err", "hits_snrs", "compton_snrs", "single_snrs", "hits_snrs_err", "compton_snrs_err",
                          "single_snrs_err", "num_offsat"]:
            #############################################################################################################
            # Filtering the satellites for some items
            #############################################################################################################
            if item in ["compton_b_rate", "mu100_ref", "mu100_err_ref", "s_eff_compton_ref", "s_eff_compton",
                        "s_eff_compton_err", "compton", "compton_cr", "df_compton"]:
              selected_sats = []
              for index_sat in considered_sats:
                if self[index_sat].const_beneficial_compton:
                  selected_sats.append(index_sat)
              selected_sats = np.array(selected_sats)
            elif item in ["single_b_rate", "s_eff_single_ref", "s_eff_single", "s_eff_single_err", "single", "single_cr", "df_single"]:
              selected_sats = []
              for index_sat in considered_sats:
                if self[index_sat].const_beneficial_single:
                  selected_sats.append(index_sat)
              selected_sats = np.array(selected_sats)
            else:
              selected_sats = considered_sats
            #############################################################################################################
            # Putting together the values
            #############################################################################################################
            #############################################################################################################
            # All the same
            #############################################################################################################
            # Values supposed to be the same for all sat and all sims so it doesn't change and is set using 1 sat
            if item in ["bins", "array_dtype"]:
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
            elif item in ["compton_b_rate", "single_b_rate", "s_eff_compton_ref", "s_eff_single_ref", "s_eff_compton",
                          "s_eff_single", "single", "single_cr", "compton", "compton_cr", "n_sat_detect"]:
              temp_val = 0
              for num_sat in selected_sats:
                temp_val += getattr(self[num_sat], item)
              setattr(self.const_data[ite_const], item, temp_val)
            #############################################################################################################
            # propagated sum
            #############################################################################################################
            # Values summed
            elif item in ["s_eff_compton_err", "s_eff_single_err"]:
              temp_val = 0
              for num_sat in selected_sats:
                temp_val += getattr(self[num_sat], item)**2
              setattr(self.const_data[ite_const], item, np.sqrt(temp_val))
            #############################################################################################################
            # Dataframe combination
            #############################################################################################################
            # Values stored in a 1D array that have to be concatenated (except unpol that needs another verification)
            elif item in ["df_compton", "df_single"]:
              temp_df_list = []
              for num_sat in selected_sats:
                temp_df_list.append(getattr(self[num_sat], item))
              if len(temp_df_list) == 0:
                setattr(self.const_data[ite_const], item, getattr(self[selected_sats[0]], item))
              else:
                setattr(self.const_data[ite_const], item, pd.concat(temp_df_list, ignore_index=True))
            #############################################################################################################
            # Weighted mean
            #############################################################################################################
            # mu100_ref key
            elif item in ["mu100_ref"]:
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
            # mu100 err
            #############################################################################################################
            # mu100_err_ref key
            elif item in ["mu100_err_ref"]:
              somme_ev = 0
              somme_ev_mu = 0
              ev_mu_err = []
              mu = []
              ev = []
              for num_sat in selected_sats:
                somme_ev += self[num_sat].compton
                somme_ev_mu += self[num_sat].mu100_ref * self[num_sat].compton
                ev_mu_err.append(getattr(self[num_sat], item) * self[num_sat].compton)
                mu.append(self[num_sat].mu100_ref)
                ev.append(self[num_sat].compton)
              ev_mu_err = np.array(ev_mu_err)
              mu = np.array(mu)
              ev = np.array(ev)

              err_val = np.sqrt(np.sum((ev_mu_err / somme_ev)**2 + ev * ((mu * somme_ev - somme_ev_mu) / somme_ev**2)**2))
              if somme_ev != 0:
                setattr(self.const_data[ite_const], item, err_val)
              else:
                setattr(self.const_data[ite_const], item, 0)
            #############################################################################################################
            # Appened
            #############################################################################################################
            elif item in ["num_sat", "const_beneficial_compton", "const_beneficial_single", "bkg_index", "sat_mag_dec", "mu_index"]:
              temp_list = []
              for num_sat in selected_sats:
                temp_list.append(getattr(self[num_sat], item))
              setattr(self.const_data[ite_const], item, np.array(temp_list))
            #############################################################################################################
            # List sum
            #############################################################################################################
            elif item in ["const_beneficial_trigger_4s", "const_beneficial_trigger_3s", "const_beneficial_trigger_2s", "const_beneficial_trigger_1s"]:
              temp_list = np.zeros(9, dtype=np.int16)
              for num_sat in selected_sats:
                temp_list += getattr(self[num_sat], item)
              setattr(self.const_data[ite_const], item, temp_list)
            else:
              raise AttributeError(f"Item '{item}' not found")

  def make_condensed_const(self, num_of_down_per_const, off_sats, const=None, dysfunction_enabled=True):
    """
    Creates a one or more constellations of several satellites by putting together only the important results
        Used when the data container for the constellation is condensed (essential data)
    :param num_of_down_per_const: List containing the number of satellite not working in the constellations (taken randomly)
        Gives the number of constellations wanted and their number of off satellites
        Example : if only one constellation is wanted with 2 satellites not working  num_of_down_per_const = [2]
        Example : if 3 constellation are wanted with 0, 2 and 5 satellites not working  num_of_down_per_const = [0, 2, 5]
    :param off_sats: list containing the indexes of satellites not working
    :param const: array with the indexes of the satellites to put in the constellation
        If None all satellites are considered
    :param dysfunction_enabled: True if dysfunction of satellites is taken into account (the value of the ints in num_of_down_per_const)
    """
    if const is None:
      const = np.array(range(self.n_sat))
    ###################################################################################################################
    # Required for usual constellation
    ###################################################################################################################
    in_sight_sat = np.where(np.array(self) == None, False, True)
    ###################################################################################################################
    # Preparing which satellite data will be combined
    ###################################################################################################################
    list_considered_sat = []
    self.const_data = []
    if dysfunction_enabled:
      number_const = len(num_of_down_per_const)
    else:
      number_const = 1
    for const_ite in range(number_const):
      if off_sats[const_ite] is None:
        self.const_data.append(ConstData(0))
      elif type(off_sats[const_ite]) is list:
        self.const_data.append(ConstData(len(off_sats[const_ite])))
      else:
        raise TypeError("Error while making the constellations : the off_sats variable must be None or a list of index")
      in_sight_temp = in_sight_sat
      if off_sats[const_ite] is not None:
        for index in off_sats[const_ite]:
          in_sight_temp[index] = False
      sat_considered_temp = const[in_sight_temp]
      list_considered_sat.append(sat_considered_temp)
    ###################################################################################################################
    # Combining satellite data
    ###################################################################################################################
    for ite_const, considered_sats in enumerate(list_considered_sat):
      if len(considered_sats) == 0:
        self.const_data[ite_const] = None
      else:
        for item in self.const_data[ite_const].__dict__.keys():
          ###############################################################################################################
          # Not changed
          ###############################################################################################################
          # The fieldselected here stay as they are with their basic initialisation (most of the time None)
          if item not in ["mdp", "mdp_err", "hits_snrs", "compton_snrs", "single_snrs", "hits_snrs_err", "compton_snrs_err",
                          "single_snrs_err", "num_offsat"]:
            #############################################################################################################
            # Filtering the satellites for some items
            #############################################################################################################
            if item in ["compton_b_rate", "mu100_ref", "mu100_err_ref", "compton_time", "s_eff_compton_ref",
                        "s_eff_compton", "s_eff_compton_err", "compton", "compton_cr"]:
              selected_sats = []
              for index_sat in considered_sats:
                if self[index_sat].const_beneficial_compton:
                  selected_sats.append(index_sat)
              selected_sats = np.array(selected_sats)
            elif item in ["single_b_rate", "s_eff_single_ref", "single_time", "s_eff_single", "s_eff_single_err", "single", "single_cr"]:
              selected_sats = []
              for index_sat in considered_sats:
                if self[index_sat].const_beneficial_single:
                  selected_sats.append(index_sat)
              selected_sats = np.array(selected_sats)
            else:
              selected_sats = considered_sats
            #############################################################################################################
            # Putting together the values
            #############################################################################################################
            #############################################################################################################
            # Summed
            #############################################################################################################
            # Values summed
            if item in ["compton_b_rate", "single_b_rate", "s_eff_compton_ref", "s_eff_single_ref", "s_eff_compton",
                        "s_eff_single", "single", "single_cr", "compton", "compton_cr", "n_sat_detect"]:
              temp_val = 0
              for num_sat in selected_sats:
                temp_val += getattr(self[num_sat], item)
              setattr(self.const_data[ite_const], item, temp_val)
            #############################################################################################################
            # propagated sum
            #############################################################################################################
            # Values summed
            elif item in ["s_eff_compton_err", "s_eff_single_err"]:
              temp_val = 0
              for num_sat in selected_sats:
                temp_val += getattr(self[num_sat], item) ** 2
              setattr(self.const_data[ite_const], item, np.sqrt(temp_val))
            #############################################################################################################
            # compton and single times using the dataframes
            #############################################################################################################
            elif item in ["compton_time"]:
              temp_array = np.array([])
              for num_sat in selected_sats:
                temp_array = np.concatenate((temp_array, getattr(self[num_sat], "df_compton").compton_time.values))
              setattr(self.const_data[ite_const], "compton_time", temp_array)
            elif item in ["single_time"]:
              temp_array = np.array([])
              for num_sat in selected_sats:
                temp_array = np.concatenate((temp_array, getattr(self[num_sat], "df_single").single_time.values))
              setattr(self.const_data[ite_const], "single_time", temp_array)
            #############################################################################################################
            # Weighted mean
            #############################################################################################################
            # mu100_ref key
            elif item in ["mu100_ref"]:
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
            # mu100 err
            #############################################################################################################
            # mu100_err_ref key
            elif item in ["mu100_err_ref"]:
              somme_ev = 0
              somme_ev_mu = 0
              ev_mu_err = []
              mu = []
              ev = []
              for num_sat in selected_sats:
                somme_ev += self[num_sat].compton
                somme_ev_mu += self[num_sat].mu100_ref * self[num_sat].compton
                ev_mu_err.append(getattr(self[num_sat], item) * self[num_sat].compton)
                mu.append(self[num_sat].mu100_ref)
                ev.append(self[num_sat].compton)
              ev_mu_err = np.array(ev_mu_err)
              mu = np.array(mu)
              ev = np.array(ev)

              err_val = np.sqrt(np.sum((ev_mu_err / somme_ev)**2 + ev * ((mu * somme_ev - somme_ev_mu) / somme_ev**2)**2))
              if somme_ev != 0:
                setattr(self.const_data[ite_const], item, err_val)
              else:
                setattr(self.const_data[ite_const], item, 0)
            #############################################################################################################
            # Appened
            #############################################################################################################
            elif item in ["num_sat", "const_beneficial_compton", "const_beneficial_single", "bkg_index", "sat_mag_dec", "mu_index"]:
              temp_list = []
              for num_sat in selected_sats:
                temp_list.append(getattr(self[num_sat], item))
              setattr(self.const_data[ite_const], item, np.array(temp_list))
            #############################################################################################################
            # List sum
            #############################################################################################################
            elif item in ["const_beneficial_trigger_4s", "const_beneficial_trigger_3s", "const_beneficial_trigger_2s", "const_beneficial_trigger_1s"]:
              temp_list = np.zeros(9, dtype=np.int16)
              for num_sat in selected_sats:
                temp_list += getattr(self[num_sat], item)
              setattr(self.const_data[ite_const], item, temp_list)
            else:
              raise AttributeError(f"Item '{item}' not found")
