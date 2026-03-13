# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  MAllSimData.py
# Class to contain GRB data for a given source from all simulations of the source
# ================================================================

# Package imports
import numpy as np

# Developped modules imports
from src.General.funcmod import calc_flux_sample, calc_flux_gbm
from src.Analysis.MAllSatData import AllSatData


class AllSimData(list):
  """
  Class containing all the data for 1 GRB (or other source) for a full set of trafiles
  """
  def __init__(self, all_sim_data, source_ite, cat_data, sat_info, param_sim_duration, bkgdata, mudata, options):
    """
    :param all_sim_data: 2D list containing simulation filenames
    :param source_ite: Catalogue iteration/index of the source simulated
    :param cat_data: Catalogue used (usually GBM or synthetic GRB catalogue)
    :param sat_info: Orbital information on the satellites
    :param param_sim_duration: duration of the simulation (fixed of t90)
    :param bkgdata: Background data container
    :param mudata: mu100/Seff data container
    :param options: List containing [erg_cut, armcut, geometry, init_correction, polarigram_bins]
    options gives information about the following options, required for data analysis
        ergcut: Energy cut to use
        armcut: ARM (Angular Resolution Measurement) cut to use
        geometry: geometry of the mass model used for the simulation
        init_correction: True if the polarigrams should be corrected (useful when adding them together for the constellation)
        polarigram_bins: Bins for the polarigram
    """
    temp_list = []
    self.n_sim_det = 0
    if cat_data.cat_type == "GBM":
      self.source_name = cat_data.df.name.values[source_ite]
      self.redshift = None
      self.source_duration = cat_data.df.t90.values[source_ite]
      # Retrieving pflux and mean flux : the photon flux at the peak flux (or mean photon flux) of the burst [photons/cm2/s]
      self.best_fit_model = cat_data.df.flnc_best_fitting_model.values[source_ite]
      self.best_fit_mean_flux = cat_data.df.mean_flux.values[source_ite]
      self.best_fit_p_flux = cat_data.df.peak_flux.values[source_ite]
      self.ergcut_mean_flux = calc_flux_gbm(cat_data, source_ite, options[0])
      if self.best_fit_p_flux is not None:
        self.ergcut_peak_flux = self.best_fit_p_flux * self.ergcut_mean_flux / self.best_fit_mean_flux
      else:
        self.ergcut_peak_flux = None
      # Retrieving fluence of the source [photons/cm²]
      self.source_fluence = self.ergcut_mean_flux * self.source_duration
      # Retrieving energy fluence of the source [erg/cm²]
      self.source_energy_fluence = cat_data.df.fluence.values[source_ite]
    elif cat_data.cat_type == "sampled":
      self.source_name = cat_data.df.name.values[source_ite]
      self.redshift = cat_data.df.z_obs.values[source_ite]
      self.source_duration = float(cat_data.df.t90.values[source_ite])
      self.best_fit_model = "band"
      self.best_fit_mean_flux = float(cat_data.df.mean_flux.values[source_ite])
      self.best_fit_p_flux = float(cat_data.df.peak_flux.values[source_ite])
      self.ergcut_peak_flux = calc_flux_sample(cat_data, source_ite, options[0])
      self.ergcut_mean_flux = self.best_fit_mean_flux * self.ergcut_peak_flux / self.best_fit_p_flux
      self.source_fluence = self.ergcut_mean_flux * self.source_duration
      self.source_energy_fluence = None
    else:
      raise ValueError("Wrong catalog type")
    if param_sim_duration.isdigit():
      sim_duration = float(param_sim_duration)
    elif param_sim_duration == "t90" or param_sim_duration == "lc":
      sim_duration = self.source_duration
    else:
      sim_duration = None
      print("Warning : unusual sim duration, please check the parameter file.")

    output_message = f"{np.count_nonzero(np.array(all_sim_data).flatten() != None)} files to be loaded for source {self.source_name} : "
    for sim_ite, all_sat_data in enumerate(all_sim_data):
      not_none = len(all_sat_data) - all_sat_data.count(None)
      output_message += f"\n  Total of {not_none} files loaded for simulation {sim_ite}"
      if not_none != 0:
        self.n_sim_det += 1
      temp_list.append(AllSatData(all_sat_data, sat_info, sim_duration, [self.source_duration, self.source_fluence], bkgdata, mudata, options))
    print(output_message)

    list.__init__(self, temp_list)
