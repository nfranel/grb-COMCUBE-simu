# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import subprocess
# Developped modules imports
from funcmod import *
from MAllSatData import AllSatData


class AllSimData(list):
  """
  Class containing all the data for 1 GRB (or other source) for a full set of trafiles
  """

  def __init__(self, sim_prefix, source_ite, cat_data, n_sim, sat_info, param_sim_duration, bkg_data, mu_data, options):
    temp_list = []
    self.n_sim_det = 0
    if type(cat_data) is list:
      self.source_name = cat_data[0][source_ite]
      self.source_duration = float(cat_data[1][source_ite])
      self.best_fit_model = None
      self.best_fit_p_flux = None
      self.best_fit_mean_flux = None
      self.source_fluence = None
    else:
      self.source_name = cat_data.name[source_ite]
      self.source_duration = float(cat_data.t90[source_ite])
      # Retrieving pflux and mean flux : the photon flux at the peak flux (or mean photon flux) of the burst [photons/cm2/s]
      self.best_fit_model = getattr(cat_data, "flnc_best_fitting_model")[source_ite].rstrip()
      self.best_fit_p_flux = float(getattr(cat_data, f"{getattr(cat_data, 'pflx_best_fitting_model')[source_ite].rstrip()}_phtflux")[source_ite])
      self.best_fit_mean_flux = float(getattr(cat_data, f"{getattr(cat_data, 'flnc_best_fitting_model')[source_ite].rstrip()}_phtflux")[source_ite])
      # Retrieving fluence of the source [photons/cm2]
      self.source_fluence = calc_fluence(cat_data, source_ite, options[-1]) * self.source_duration
      # print("pflx", self.best_fit_p_flux)
      # print(self.best_fit_mean_flux, "  --  " , self.best_fit_mean_flux * self.source_duration, "  -  ", self.source_fluence)
    if param_sim_duration.isdigit():
      sim_duration = float(param_sim_duration)
    elif param_sim_duration == "t90":
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
          temp_list.append(AllSatData(source_prefix, num_sim, sat_info, sim_duration, self.source_fluence, bkg_data, mu_data, options))
          self.n_sim_det += 1

    list.__init__(self, temp_list)
    for sim_ite, sim in enumerate(self):
      if sim is not None:
        if output_message is not None:
          output_message += f"\n  Total of {sim.loading_count} files loaded for simulation {sim_ite}"
    print(output_message)

  @staticmethod
  def get_keys():
    print("======================================================================")
    print("    Attributes")
    print(" Number of simulations detected by the constellation : .n_sim_det")
    print(" Name of the source :                                  .source_name")
    print(" Duration of the source (t90 for GRB) :                .source_duration")
    print(" ===== Attribute that needs to be handled + 2 cases (full FoV or full sky)")
    print(" Probability of having a detection                     .proba_detec")
    print(" Probability of being able to construct an image :     .proba_compton_image")
    print("======================================================================")
    print("    Methods")
    print("======================================================================")

  def set_probabilities(self, n_sat, snr_min=5, n_image_min=50):
    """
    Calculates detection probability and probability of having a correct compton image
    """
    temp_single_proba_detec = np.zeros(n_sat)
    temp_compton_proba_detec = np.zeros(n_sat)
    temp_proba_compton_image = np.zeros(n_sat)
    temp_const_single_proba_detec = 0
    temp_const_compton_proba_detec = 0
    temp_const_proba_compton_image = 0
    for sim in self:
      if sim is not None:
        for sat_ite, sat in enumerate(sim):
          if sat is not None:
            if sat.snr_single_t90 >= snr_min:
              temp_single_proba_detec[sat_ite] += 1
            if sat.snr_compton_t90 >= snr_min:
              temp_compton_proba_detec[sat_ite] += 1
            if sat.compton >= n_image_min:
              temp_proba_compton_image[sat_ite] += 1
        if sim.const_data.snr_single_t90 >= snr_min:
          temp_const_single_proba_detec += 1
        if sim.const_data.snr_compton_t90 >= snr_min:
          temp_const_compton_proba_detec += 1
        if sim.const_data.compton >= n_image_min:
          temp_const_proba_compton_image += 1

    if self.n_sim_det != 0:
      self.proba_single_detec_fov = temp_single_proba_detec / self.n_sim_det
      self.proba_compton_detec_fov = temp_compton_proba_detec / self.n_sim_det
      self.proba_compton_image_fov = temp_proba_compton_image / self.n_sim_det
      self.const_single_proba_detec_fov = temp_const_single_proba_detec / self.n_sim_det
      self.const_compton_proba_detec_fov = temp_const_compton_proba_detec / self.n_sim_det
      self.const_proba_compton_image_fov = temp_const_proba_compton_image / self.n_sim_det
    else:
      self.proba_single_detec_fov = 0
      self.proba_compton_detec_fov = 0
      self.proba_compton_image_fov = 0
      self.const_single_proba_detec_fov = 0
      self.const_compton_proba_detec_fov = 0
      self.const_proba_compton_image_fov = 0
    if len(self) != 0:
      self.proba_single_detec_sky = temp_single_proba_detec / len(self)
      self.proba_compton_detec_sky = temp_compton_proba_detec / len(self)
      self.proba_compton_image_sky = temp_proba_compton_image / len(self)
      self.const_single_proba_detec_sky = temp_const_single_proba_detec / len(self)
      self.const_compton_proba_detec_sky = temp_const_compton_proba_detec / len(self)
      self.const_proba_compton_image_sky = temp_const_proba_compton_image / len(self)
    else:
      self.proba_single_detec_sky = 0
      self.proba_compton_detec_sky = 0
      self.proba_compton_image_sky = 0
      self.const_single_proba_detec_sky = 0
      self.const_compton_proba_detec_sky = 0
      self.const_proba_compton_image_sky = 0
