# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  polarization_class.py
# Contains a function and a class to contain polarisation calculation using the models from Toma et al. 2009 and Pearce et al. 2019
# ================================================================

# Regular imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from itertools import repeat

# Developped modules imports
from src.General.funcmod import arg_convert, var_ite_setting, values_number, SO_PF_distri, SR_PF_distri, CD_PF_distri, PJ_PF_distri, acc_reject, gauss
from src.Polarization.models import integral_calculation_SO, integral_calculation_SR, integral_calculation_CD, integral_calculation_PJ

import matplotlib as mpl
mpl.use('Qt5Agg')


class PolVSAngleRatio:
    """
    Computes and stores the polarization fraction (PF) as a function of the
    viewing-angle-to-jet-opening-angle ratio q = theta_nu / theta_j for one of
    four GRB emission models: SO (Synchrotron Ordered), SR (Synchrotron Random),
    CD (Compton Drag), or PJ (Patchy Jet). Each parameter can be supplied either
    as a fixed scalar or as a distribution descriptor tuple that triggers random
    sampling. The calculation is parallelised via multiprocessing. Results are
    stored in self.data_df as a pandas DataFrame.
    """
    def __init__(self, model=None, gamma_range=None, red_z_range=None, theta_j_range=None, theta_nu_range=None, nu_0_range=None,
                 alpha_range=None, beta_range=None, nu_min=None, nu_max=None, jet_model=None, flux_rejection=False, integ_steps=None,
                 confidence=None, parallel="all"):
        """
        Instantiates a PolVSAngleRatio object, sets all parameter ranges to their
        defaults if not provided, and immediately runs the full polarization
        fraction calculation pipeline via show_parameters() and pf_run().
        :param model: str or None, emission model to use; one of "SO", "SR", "CD", or "PJ"; default="SO"
        :param gamma_range: float or tuple or None, Lorentz factor value or distribution descriptor; default=100
        :param red_z_range: float or tuple or None, redshift value or distribution descriptor; default=('distri', 3)
        :param theta_j_range: float or tuple or None, jet half-opening angle value or distribution descriptor [rad]; default=('distri', 3)
        :param theta_nu_range: float or tuple or str or None, viewing angle value, distribution descriptor, or "toma_curve"
            to span (0.001*theta_j, 5*theta_j) [rad]; default=('distri', 3)
        :param nu_0_range: float or tuple or None, reference frequency value or distribution descriptor [keV]; default=3.5
        :param alpha_range: float or tuple or None, Band low-energy spectral index value or distribution descriptor; default=-0.8
        :param beta_range: float or tuple or None, Band high-energy spectral index value or distribution descriptor; default=-2.2
        :param nu_min: float or None, lower bound of the frequency integration range [keV]; default=60
        :param nu_max: float or None, upper bound of the frequency integration range [keV]; default=500
        :param jet_model: str or None, jet geometry model; default="top-hat"
        :param flux_rejection: bool, if True applies flux-based acceptance rejection when building the parameter list; default=False
        :param integ_steps: int or None, number of Monte Carlo sample points per integration dimension; default=70
        :param confidence: float or None, z-score multiplier for the error calculation; default=1.96
        :param parallel: int or str, number of worker processes; use "all" to use all available CPUs, or an int > 1 for a fixed pool size,
            or any other value for single-threaded execution; default="all"
        """
        if model is None:
            model = "SO"
        if theta_j_range is None:
            theta_j_range = ('distri', 3)
        if theta_nu_range is None:
            theta_nu_range = ('distri', 3)
        if red_z_range is None:
            red_z_range = ('distri', 3)
        if gamma_range is None:
            gamma_range = 100
        if nu_0_range is None:
            nu_0_range = 3.5
        if alpha_range is None:
            alpha_range = -0.8
        if beta_range is None:
            beta_range = -2.2
        if nu_min is None:
            nu_min = 60
        if nu_max is None:
            nu_max = 500
        if jet_model is None:
            jet_model = "top-hat"
        if integ_steps is None:
            integ_steps = 70
        if confidence is None:
            confidence = 1.96

        # Attributes used in formula
        self.model = model
        self.theta_j_range = theta_j_range
        self.theta_nu_range = theta_nu_range
        self.red_z_range = red_z_range
        self.gamma_range = gamma_range
        self.nu_0_range = nu_0_range
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.jet_model = jet_model
        self.flux_rejection = flux_rejection
        self.top_hat_max_beaming = 1
        self.structured_max_beaming = 3

        # Data attributes
        self.columns = ["pf", "error_pf", "gamma", "z", "theta_j", "theta_nu", "nu_0", "alpha", "beta", "q", "yj", "gamma_nu_0"]
        self.data_df = None

        # Other attributes
        self.integ_steps = integ_steps
        self.confidence = confidence
        self.parallel = parallel

        self.show_parameters()
        self.pf_run()

    def create_params(self):
        """
        Builds the full list of parameter tuples to be evaluated, by iterating over all combinations of the ranges stored in the instance attributes.
        Each tuple encodes one (gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta) point together with its iteration index and jet-model settings.
        :returns: np.ndarray, 2-D array of shape (N_combinations, n_fields) where each row is one parameter set as returned by var_ite_setting()
        """
        print("Creation of parameters list")
        init_time = time.time()
        arg_list = []
        ite_count = 0
        if self.model == "PJ":
          opening_factor = self.structured_max_beaming
        else:
          opening_factor = self.top_hat_max_beaming
        # In that case we use a list for theta j and a 3-tuple for theta nu
        for gamma, ite_gamma in arg_convert(self.gamma_range):
            for red_z, ite_red_z in arg_convert(self.red_z_range):
                for theta_j, ite_theta_j in arg_convert(self.theta_j_range):
                    if self.theta_nu_range == "toma_curve":
                        theta_nu_selec = (0.001*theta_j, 5*theta_j, 1000)
                    else:
                        theta_nu_selec = self.theta_nu_range
                    for theta_nu, ite_theta_nu in arg_convert(theta_nu_selec):
                        for nu_0, ite_nu_0 in arg_convert(self.nu_0_range):
                            for alpha, ite_alpha in arg_convert(self.alpha_range):
                                for beta, ite_beta in arg_convert(self.beta_range):
                                    arg_list.append(var_ite_setting(ite_count, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, opening_factor, self.jet_model, self.flux_rejection))
                                    ite_count += 1
        arg_list = np.array(arg_list)
        print(f"Creation finished after {time.time()-init_time} seconds")
        return arg_list

    def integral_calculation(self, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta):
        """
        Dispatches the polarization fraction calculation to the appropriate model function defined in models.py,
        using the integration settings stored in self.integ_steps and self.confidence.
        :param gamma: float, Lorentz factor of the jet
        :param red_z: float, source redshift
        :param theta_j: float, jet half-opening angle [rad]
        :param theta_nu: float, off-axis viewing angle [rad]
        :param nu_0: float, reference frequency [keV]
        :param alpha: float, Band low-energy spectral index
        :param beta: float, Band high-energy spectral index
        :returns: float, float, the polarization fraction and its confidence-interval half-width
            (model-specific return values are truncated to these two quantities)
        :raises ValueError: if self.model is not one of "SO", "SR", "CD", "PJ"
        """
        if self.model == "SO":
            return integral_calculation_SO(self.nu_min, self.nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=self.integ_steps, confidence=self.confidence)[:2]
        elif self.model == "SR":
            return integral_calculation_SR(self.nu_min, self.nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=self.integ_steps, confidence=self.confidence)
        elif self.model == "CD":
            return integral_calculation_CD(self.nu_min, self.nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=self.integ_steps, confidence=self.confidence)
        elif self.model == "PJ":
            return integral_calculation_PJ(gamma, theta_j, theta_nu)
        else:
          raise ValueError

    def pf_calculation(self, param_list, timer_info=None):
        """
        Computes the polarization fraction for a single parameter set and optionally prints a running-time
        estimate for the first two completed iterations (iteration 0 and iteration 300).
        :param param_list: np.ndarray or list, row from the parameter matrix produced by create_params()
        :param timer_info: list or None, two-element list [start_time, total_count]
            used to estimate remaining run time; if None no timing output is printed, default=None
        :returns: list, nine-element list: [pf, pf_error, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta]
        """
        init_time = time.time()
        pf_val, pf_error = self.integral_calculation(param_list[1], param_list[2], param_list[3], param_list[4], param_list[5],
                                                     param_list[6], param_list[7])
        if timer_info is not None:
            if int(param_list[0]) == 0:
                print(f"Initial time is : {time.strftime('%H:%M:%S', time.localtime())}")
                est_time = (time.time() - init_time) * timer_info[1]
                if est_time > 3600:
                    print(f"Estimated running time : {int(est_time / 36) / 100} h")
                elif est_time > 60:
                    print(f"Estimated running time : {int(est_time / 6 * 10) / 100} min")
                else:
                    print(f"Estimated running time : {int(est_time)} s")
            elif int(param_list[0]) == 300:
                # Estimation of the impact of the threading
                time_one = (time.time() - init_time)
                time_multiple = (time.time() - timer_info[0])
                threading_factor = 300 /time_multiple * time_one
                est_time = time_one * timer_info[1] / threading_factor
                if est_time > 3600:
                    print(f"Second estimated running time : {int(est_time / 36) / 100} h")
                elif est_time > 60:
                    print(f"Second estimated running time : {int(est_time / 6 * 10) / 100} min")
                else:
                    print(f"Second estimated running time : {int(est_time)} s")
        return [pf_val, pf_error, param_list[1], param_list[2], param_list[3], param_list[4], param_list[5],
                param_list[6], param_list[7]]

    def show_parameters(self):
        """
        Prints a formatted summary of all integration and physical parameters currently set on the instance to stdout.
        """
        print("======================================================================================")
        print("Parameter used for the PF estimation")
        print("======================================================================================")
        print(f"Number of iterations for calculating the integrals : int_step = {self.integ_steps}")
        print(f"Confidence used for calculating the error : confidence = {self.confidence} sigma")
        print("======================================================================================")
        print(f"Parameters used for the simulations and the number of values if a distribution is used")
        print(f"Values for gamma = {self.gamma_range}")
        print(f"Values for redshift = {self.red_z_range}")
        print(f"Values for theta j = {self.theta_j_range}")
        print(f"Values for theta nu = {self.theta_nu_range}")
        print(f"Values for nu 0 = {self.nu_0_range}")
        print(f"Values for alpha = {self.alpha_range}")
        print(f"Values for beta = {self.beta_range}")
        print("======================================================================================")

    def pf_run(self, loading_time=True):
        """
        Orchestrates the full polarization fraction computation: builds the parameter matrix via create_params(),
        dispatches pf_calculation() over all rows (in parallel if self.parallel > 1 or "all"), and stores the
        results in self.data_df together with the derived columns q, yj, and gamma_nu_0.
        :param loading_time: bool, if True passes timing information to pf_calculation() so that run-time estimates
            are printed after the first two completed iterations; default=True
        """
        print(f"=========================================\nRun starting for model {self.model}\n=========================================")
        numb_val = values_number(self.gamma_range, self.red_z_range, self.theta_j_range, self.theta_nu_range, self.nu_0_range,
                                 self.alpha_range, self.beta_range)
        # self.data_df = VarContainer(numb_val)
        param_matrix = self.create_params()
        if loading_time:
            init_time = time.time()
            timer_var = [init_time, numb_val]
        else:
            timer_var = None

        if self.parallel == 'all':
            print("Parallel calculation of polarization fractions with all threads")
            with mp.Pool() as pool:
                data_to_save = pool.starmap(self.pf_calculation, zip(param_matrix, repeat(timer_var)))
        elif type(self.parallel) is int and self.parallel > 1:
            print(f"Parallel calculation of polarization fractions with {self.parallel} threads")
            with mp.Pool(self.parallel) as pool:
                data_to_save = pool.starmap(self.pf_calculation, zip(param_matrix, repeat(timer_var)))
        else:
            print(f"Parallel calculation of polarization fractions with 1 thread")
            data_to_save = [self.pf_calculation(param_list, timer_info=timer_var) for param_list in param_matrix]
        
        # print(data_to_save)
        self.data_df = pd.DataFrame(data=data_to_save, columns=self.columns[:-3])
        self.data_df["q"] = self.data_df.theta_nu / self.data_df.theta_j
        self.data_df["yj"] = (self.data_df.theta_j * self.data_df.gamma) ** 2
        self.data_df["gamma_nu_0"] = self.data_df.gamma * self.data_df.nu_0

        print("Run finished")

    def toma_display(self):
        """
        Plots the polarization fraction as a function of q = theta_nu / theta_j for four fixed values of yj (0.1, 1, 10, 100).
        Reproduces the style of the Toma et al. (2009) figure.
        """
        if self.data_df is not None:
            colors = ["blue", "orange", "green", "red"]
            colors_err = ["lightblue", "moccasin", "lightgreen", "tomato"]
            yj_list = [0.1, 1, 10, 100]
            figure, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
            for yj_index, yj in enumerate(yj_list):
                df_select = self.data_df[np.round(self.data_df.yj, 2) == yj]
                x_list = df_select.q.values
                y_list = df_select.pf.values
                errors = df_select.error_pf.values

                ax1.plot(x_list, y_list, label=f'yj = {yj}', color=colors[yj_index])
                ax1.fill_between(x_list, y_list - errors, y_list + errors, alpha=0.4, color=colors[yj_index])

                ax1.set(xlabel=r'q=$\theta_{\nu}$/$\theta_j$', ylabel='PF',
                        title=f"Model {self.model}\n" + r'Variation of PF as fonction of q=$\theta_{\nu}$/$\theta_j$ using different values of yj',
                        ylim=(0, 1))
            ax1.legend()
            figure.show()

# USE EXAMPLE
# simtime = time.time()
# for model in ["SO", "SR", "CD", "PJ"]:
# # for model in ["SO", "SR"]:
#     test_distri = PolVSAngleRatio(model=model, gamma_range=100, red_z_range=1, theta_j_range=list(np.sqrt([0.1, 1, 10, 100]) / 100),
#                                      theta_nu_range="toma_curve", nu_0_range=3.5, alpha_range=-0.8, beta_range=-2.2,
#                                      nu_min=None, nu_max=None, integ_steps=150, confidence=1.96, parallel=10)
#     test_distri.toma_display()
# print(f"TIME TAKEN FOR 4 MODELS : {time.time() - simtime} s")


############################################################################################################################################################################
# Function to create distributions
############################################################################################################################################################################
def pol_distribution_maker(gamma_range, red_z_range, theta_j_range, theta_nu_range, nu_0_range, alpha_range, beta_range, int_step, n_distri, savecom=None):
  """
  Generates polarization fraction distributions for all four emission models (SO, SR, CD, PJ)
  by instantiating PolVSAngleRatio for each, then plots and optionally saves the results.
  Each parameter can be a fixed value or the string "distri" / a distribution-variant string to
  trigger random sampling of n_distri values; the resulting parameter tuples are built automatically.
  :param gamma_range: float or str, Lorentz factor value, or "distri" to sample n_distri values from the default gamma distribution
  :param red_z_range: float or str, redshift value, or "distri" to sample n_distri values from the default redshift distribution
  :param theta_j_range: float or str, jet half-opening angle value [rad], or "distri_toma" / "distri_lognorm" to sample n_distri values
  :param theta_nu_range: float or str, viewing angle value [rad], or "distri", "distri_pearce", or "distri_toma" to sample n_distri values
  :param nu_0_range: float or str, reference frequency value [keV], or "distri" to sample n_distri values
  :param alpha_range: float or str, Band low-energy spectral index value, or "distri" to sample n_distri values
  :param beta_range: float or str, Band high-energy spectral index value, or "distri" to sample n_distri values
  :param int_step: int, number of Monte Carlo sample points per integration dimension passed to PolVSAngleRatio
  :param n_distri: int, number of random draws when a parameter uses a distribution; the total number of PF estimates per model is n_distri**2
  :param savecom: str or None, if not None, saves the PF distribution figure as
      "../Data/Polar/{savecom}" and the underlying data to an HDF5 file at
      "../Data/Polar/{savecom}.h5"; default=None
  """
  list_distri = []
  dir = "../Data/Polar"
  mpl.use("Qt5Agg")
  if gamma_range == "distri":
    gamma_range_param = (gamma_range, n_distri)
  else:
    gamma_range_param = gamma_range
  if red_z_range == "distri":
    red_z_range_param = (red_z_range, n_distri)
  else:
    red_z_range_param = red_z_range
  if theta_j_range in ["distri_toma", "distri_lognorm"]:
    theta_j_range_param = (theta_j_range, n_distri)
  else:
    theta_j_range_param = theta_j_range
  if theta_nu_range in ["distri", "distri_pearce", "distri_toma"]:
    theta_nu_range_param = (theta_nu_range, n_distri)
  else:
    theta_nu_range_param = theta_nu_range
  if nu_0_range == "distri":
    nu_0_range_param = (nu_0_range, n_distri)
  else:
    nu_0_range_param = nu_0_range
  if alpha_range == "distri":
    alpha_range_param = (alpha_range, n_distri)
  else:
    alpha_range_param = alpha_range
  if beta_range == "distri":
    beta_range_param = (beta_range, n_distri)
  else:
    beta_range_param = beta_range
  for distname in ["SO", "SR", "CD", "PJ"]:
    # for distname in ["SO"]:
    simtime = time.time()
    list_distri.append(PolVSAngleRatio(model=distname, gamma_range=gamma_range_param, red_z_range=red_z_range_param, theta_j_range=theta_j_range_param,
                                       theta_nu_range=theta_nu_range_param, nu_0_range=nu_0_range_param, alpha_range=alpha_range_param, beta_range=beta_range_param,
                                       nu_min=None, nu_max=None, jet_model="top-hat", flux_rejection=True, integ_steps=int_step,
                                       confidence=1.96, parallel=10))
    print(f"TIME TAKEN FOR {distname} : {time.time() - simtime} s")

  bins = np.linspace(0, 0.7, 60)
  x_pearce = (bins[1:] + bins[:-1]) / 2

  labels = ["Distribution of PF for SO model", "Distribution of PF for SR model",
            "Distribution of PF for CD model", "Distribution of PF for PJ model"]
  colors = ['blue', 'red', 'green', 'orange']
  fig_comp, axes = plt.subplots(len(list_distri), 1, figsize=(20, 10), sharex="all")
  fig_comp.suptitle(f"Distribution of Polarization fraction\nInt step : {int_steps}, number of pf estimated : {n_distri ** 2}")
  if len(list_distri) == 1:
    axes = [axes]
  for ax_idx in range(len(axes)):
    axes[ax_idx].hist(list_distri[ax_idx].data_df.pf.values, bins=bins, label=labels[ax_idx],
                      weights=[1 / len(list_distri[ax_idx].data_df)] * len(list_distri[ax_idx].data_df), color=colors[ax_idx])
    axes[ax_idx].scatter(x_pearce, ylist_pearce[ax_idx] / np.sum(ylist_pearce[ax_idx]), label="Values from Pearce")
    axes[ax_idx].legend()

  axes[-1].set(xlabel='Polarization fraction', xlim=(0, 0.75), xticks=np.arange(0, 0.7, 0.15))
  plt.show()

  bins = np.linspace(0, 1, 101)
  labels = ["Distribution of PF for SO model", "Distribution of PF for SR model",
            "Distribution of PF for CD model", "Distribution of PF for PJ model"]
  colors = ['blue', 'red', 'green', 'orange']

  fig_pol, axes = plt.subplots(len(list_distri), 1, figsize=(20, 10), sharex="all")
  if len(list_distri) == 1:
    axes = [axes]
  for ax_idx in range(len(axes)):
    axes[ax_idx].hist(list_distri[ax_idx].data_df.pf.values, bins=bins, label=labels[ax_idx],
                      weights=[1 / len(list_distri[ax_idx].data_df)] * len(list_distri[ax_idx].data_df), color=colors[ax_idx])
    axes[ax_idx].legend()

  axes[-1].set(xlabel='Polarization fraction', xlim=(0, 1), xticks=np.arange(0, 1.01, 0.1))
  if savecom is not None:
    plt.savefig(f"{dir}/{savecom}")
    mod_df = pd.DataFrame({"SO":list_distri[0].data_df.pf.values, "SR":list_distri[1].data_df.pf.values, "CD":list_distri[2].data_df.pf.values, "PJ":list_distri[3].data_df.pf.values})
    with pd.HDFStore(f"{dir}/{savecom}.h5", mode="w") as fpol:
      fpol.put(f"bins", pd.Series(bins))
      fpol.put("models", mod_df)

  plt.show()

############################################################################################################################################################################
# EXAMPLE for constructing distributions
############################################################################################################################################################################
# distri_nu = "distri_pearce"
# distri_j = "distri_lognorm"
# n_distri = 10
# int_steps = 150
# com = f"v10_{n_distri**2}"
# pol_distribution_maker(gamma_range=100, red_z_range="distri", theta_j_range=distri_j, theta_nu_range=distri_nu, nu_0_range=350 / 100, alpha_range="distri", beta_range="distri",
#                        int_step=int_steps, n_distri=n_distri, savecom=com)
