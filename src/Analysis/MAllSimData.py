# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

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
    :param sim_prefix: prefix used for simulations
    :param source_ite: iteration of the source simulated
    :param cat_data: GBM catalog used
    :param n_sim: number of simulation done
    :param sat_info: orbital information on the satellites
    :param param_sim_duration: duration of the simulation (fixed of t90)
    :param bkg_data: list containing the background data
    :param mu_data: list containing the mu100 data
    :param options: options for the analysis, defined in AllSourceData
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
      # p_model = cat_data.df.pflx_best_fitting_model.values[source_ite]
      # if type(p_model) == str:
      # #   The peak flux is the one obtained after fitting the best pflux model - it is the one for the pic ! So for short GRBs it's not the one over 1s but over the peak duration
        # self.best_fit_p_flux = cat_data.df[f"{p_model}_phtflux"][source_ite]
      # else:
      #   if np.isnan(p_model):
      #     self.best_fit_p_flux = None
      #   else:
      #     raise ValueError("A value for pflx_best_fitting_model is not set properly")
      # Retrieving fluence of the source [photons/cm²]
      self.ergcut_mean_flux = calc_flux_gbm(cat_data, source_ite, options[0])
      if self.best_fit_p_flux is not None:
        self.ergcut_peak_flux = self.best_fit_p_flux * self.ergcut_mean_flux / self.best_fit_mean_flux
      else:
        self.ergcut_peak_flux = None
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
      # self.ergcut_mean_flux = calc_flux_sample(cat_data, source_ite, options[0])
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
