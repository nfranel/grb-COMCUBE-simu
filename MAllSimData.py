# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import subprocess
# Developped modules imports
from funcmod import *
from funcsample import calc_flux_sample
from MAllSatData import AllSatData


class AllSimData(list):
  """
  Class containing all the data for 1 GRB (or other source) for a full set of trafiles
  """
  def __init__(self, sim_prefix, source_ite, cat_data, n_sim, sat_info, param_sim_duration, bkg_data, mu_data, options):
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
    if type(cat_data) is list:
      self.source_name = cat_data[0][source_ite]
      self.source_duration = float(cat_data[1][source_ite])
      self.best_fit_model = None
      self.best_fit_p_flux = None
      self.best_fit_mean_flux = None
      self.source_fluence = None
      self.source_energy_fluence = None
    else:
      self.source_name = cat_data.name[source_ite]
      self.source_duration = float(cat_data.t90[source_ite])
      if cat_data.cat_type == "GBM":
        # Retrieving pflux and mean flux : the photon flux at the peak flux (or mean photon flux) of the burst [photons/cm2/s]
        self.best_fit_model = getattr(cat_data, "flnc_best_fitting_model")[source_ite].rstrip()
        if f"{getattr(cat_data, 'pflx_best_fitting_model')[source_ite].rstrip()}" == "":
          self.best_fit_p_flux = None
        else:
          self.best_fit_p_flux = float(getattr(cat_data, f"{getattr(cat_data, 'pflx_best_fitting_model')[source_ite].rstrip()}_phtflux")[source_ite])
        self.best_fit_mean_flux = float(getattr(cat_data, f"{getattr(cat_data, 'flnc_best_fitting_model')[source_ite].rstrip()}_phtflux")[source_ite])
        # Retrieving fluence of the source [photons/cm²]
        self.source_fluence = calc_flux_gbm(cat_data, source_ite, options[0]) * self.source_duration
        # Retrieving energy fluence of the source [erg/cm²]
        self.source_energy_fluence = float(cat_data.fluence[source_ite])
      elif cat_data.cat_type == "sampled":
        self.best_fit_model = "band"
        self.best_fit_p_flux = None
        self.best_fit_mean_flux = float(cat_data.mean_flux[source_ite])
        self.source_fluence = calc_flux_sample(cat_data, source_ite, options[0]) * self.source_duration
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

    # These probabilities use a lot of memory, Make it differently ?
    self.proba_single_detec_fov = None
    self.proba_compton_image_fov = None
    self.const_single_proba_detec_fov = None
    self.const_proba_compton_image_fov = None
    self.proba_single_detec_sky = None
    self.proba_compton_image_sky = None
    self.const_single_proba_detec_sky = None
    self.const_proba_compton_image_sky = None
    self.proba_compton_detec_fov = None
    self.const_compton_proba_detec_fov = None
    self.proba_compton_detec_sky = None
    self.const_compton_proba_detec_sky = None

    output_message = None
    source_prefix = f"{sim_prefix}_{self.source_name}"
    flist = subprocess.getoutput("ls {}_*".format(source_prefix)).split("\n")

    if flist[0].startswith("ls: cannot access"):
      print(f"No file to be loaded for source {self.source_name}")
    else:
      output_message = f"{len(flist)} files to be loaded for source {self.source_name} : "
    for num_sim in range(n_sim):
      flist = subprocess.getoutput("ls {}_*_{:04d}_*".format(source_prefix, num_sim)).split("\n")
      if len(flist) >= 1:
        if flist[0].startswith("ls: cannot access"):
          temp_list.append(None)
        else:
          info_source = [self.source_duration, self.source_fluence]
          temp_list.append(AllSatData(source_prefix, num_sim, sat_info, sim_duration, bkg_data, mu_data, info_source, options))
          self.n_sim_det += 1

    list.__init__(self, temp_list)

    for sim_ite, sim in enumerate(self):
      if sim is not None:
        if output_message is not None:
          output_message += f"\n  Total of {sim.loading_count} files loaded for simulation {sim_ite}"
    print(output_message)

  # todo change it
  # def set_probabilities(self, n_sat, snr_min=5, n_image_min=50):
  #   """
  #   TODO probably needs some updating, get rid of these and just make a method to print it instead of wasting memory ?
  #   Calculates detection probability and probability of having a correct compton image
  #   :param n_sat: number of satellites
  #   :param snr_min: minimum snr to consider a detection
  #   :param n_image_min: minimum number of compton event required to reconstruct a compton image
  #   """
  #   temp_single_proba_detec = np.zeros(n_sat)
  #   temp_compton_proba_detec = np.zeros(n_sat)
  #   temp_proba_compton_image = np.zeros(n_sat)
  #   temp_const_single_proba_detec = 0
  #   temp_const_compton_proba_detec = 0
  #   temp_const_proba_compton_image = 0
  #   for sim in self:
  #     if sim is not None:
  #       for sat_ite, sat in enumerate(sim):
  #         if sat is not None:
  #           if sat.snr_single_t90 >= snr_min:
  #             temp_single_proba_detec[sat_ite] += 1
  #           if sat.snr_compton_t90 >= snr_min:
  #             temp_compton_proba_detec[sat_ite] += 1
  #           if sat.compton >= n_image_min:
  #             temp_proba_compton_image[sat_ite] += 1
  #       if sim.const_data.snr_single_t90 >= snr_min:
  #         temp_const_single_proba_detec += 1
  #       if sim.const_data.snr_compton_t90 >= snr_min:
  #         temp_const_compton_proba_detec += 1
  #       if sim.const_data.compton >= n_image_min:
  #         temp_const_proba_compton_image += 1
  #
  #   if self.n_sim_det != 0:
  #     self.proba_single_detec_fov = temp_single_proba_detec / self.n_sim_det
  #     self.proba_compton_detec_fov = temp_compton_proba_detec / self.n_sim_det
  #     self.proba_compton_image_fov = temp_proba_compton_image / self.n_sim_det
  #     self.const_single_proba_detec_fov = temp_const_single_proba_detec / self.n_sim_det
  #     self.const_compton_proba_detec_fov = temp_const_compton_proba_detec / self.n_sim_det
  #     self.const_proba_compton_image_fov = temp_const_proba_compton_image / self.n_sim_det
  #   else:
  #     self.proba_single_detec_fov = 0
  #     self.proba_compton_detec_fov = 0
  #     self.proba_compton_image_fov = 0
  #     self.const_single_proba_detec_fov = 0
  #     self.const_compton_proba_detec_fov = 0
  #     self.const_proba_compton_image_fov = 0
  #   if len(self) != 0:
  #     self.proba_single_detec_sky = temp_single_proba_detec / len(self)
  #     self.proba_compton_detec_sky = temp_compton_proba_detec / len(self)
  #     self.proba_compton_image_sky = temp_proba_compton_image / len(self)
  #     self.const_single_proba_detec_sky = temp_const_single_proba_detec / len(self)
  #     self.const_compton_proba_detec_sky = temp_const_compton_proba_detec / len(self)
  #     self.const_proba_compton_image_sky = temp_const_proba_compton_image / len(self)
  #   else:
  #     self.proba_single_detec_sky = 0
  #     self.proba_compton_detec_sky = 0
  #     self.proba_compton_image_sky = 0
  #     self.const_single_proba_detec_sky = 0
  #     self.const_compton_proba_detec_sky = 0
  #     self.const_proba_compton_image_sky = 0
