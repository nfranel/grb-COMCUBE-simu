# from distributions import *
# from scipy.integrate import quad
import numpy as np
from funcsample import acc_reject, generator_theta_nu, generator_nu_0, distrib_z, distrib_theta_j, distrib_theta_nu_toma, distrib_alpha, distrib_beta
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad, simpson, IntegrationWarning
import warnings

# # ########################################################################################################################
# # # Constants
# # ########################################################################################################################
# speed_c = 299792.458 # [km/s]
# astronomical_unit = 149597870700 # [m]
# parsec = astronomical_unit / np.tan(np.deg2rad(1 / 3600)) # [m]
# mpc_to_cm = 1e6 * parsec * 100 # [cm]
# elec_charge = 1.602176634e-19 # [C or J]
# erg_to_kev = 1e-10 / elec_charge # [multiply erg by it to obtain keV, divide for the inverse effect]
########################################################################################################################
# Used cosmology (given by Toma et al)
########################################################################################################################
cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

########################################################################################################################
# Functions used for the gestion of the argument values                                                                #
########################################################################################################################
def arg_convert(arg):
    if type(arg) == int or type(arg) == float:
        return zip([arg], [0])
    elif type(arg) == tuple and len(arg) == 3:
        return zip(np.linspace(arg[0], arg[1], arg[2]), range(arg[2]))
    elif type(arg) == tuple and len(arg) == 2 and arg[0].startswith("distri"):
        return zip([arg[0]] * arg[1], range(arg[1]))
    elif type(arg) == list:
        return zip(arg, range(len(arg)))
    else:
        print("Error : at least one argument doesn't have the required format")


def values_number(gamma_range_func, red_z_range_func, theta_j_range_func, theta_nu_range_func, nu_0_range_func, alpha_range_func,
                  beta_range_func):
    range_list = [theta_j_range_func, theta_nu_range_func, red_z_range_func, gamma_range_func, nu_0_range_func, alpha_range_func,
                  beta_range_func]
    values_num = 1
    for ranges in range_list:
        if type(ranges) == tuple:
            values_num = values_num * int(ranges[-1])
        elif type(ranges) == list:
            values_num = values_num * len(ranges)
    return values_num


# def friendmann_function(z, o_r, o_m, o_k, o_l):
#     """
#     Function based on friedmann's equation, to calculate the luminosity distance
#     """
#     return 1 / np.sqrt(o_r * (1 + z)**4 + o_m * (1 + z)**3 + o_k * (1 + z)**2 + o_l)


# def luminosity_distance_calc(z, o_r, o_m, o_k, o_l, h0):
#     """
#     Function to calculate the luminosity distance in a FLRW metric
#     dl = (1+z) * c / H0 * integ 0->z [friedmann's function]
#     """
#     return (1 + z) * speed_c / h0 * quad(friendmann_function, 0, z, args=(o_r, o_m, o_k, o_l))[0]


def var_ite_setting(iteration, gamma_func, red_z_func, theta_j_func, theta_nu_func, nu_0_func, alpha_func, beta_func, jet_model,
                    flux_rejection):
    """
    Function to obtain a set of parameters according to simulation settings and distributions
    """
    # Best parameters : 1.5e-8 for PJ
    #
    limit_flux = 3.5e-7  # erg/cm2/s
    calculated_flux = 0
    while calculated_flux < limit_flux:
        if gamma_func == "distri":
            print("No distri added for gamma yet, value 100 is taken")
            gamma_loop_func = 100
        else:
            gamma_loop_func = gamma_func

        if red_z_func == "distri":
            red_z_loop_func = acc_reject(distrib_z, [], 0, 10)
        else:
            red_z_loop_func = red_z_func

        if theta_j_func == "distri":
            # Correction term from Yonetoku, to make the opening half angle independent of the redshift
            # Used to obtain the distrib at any redshift using the distrib at redshift z=1
            # Not sure it is usefull, it depends if it's already taken into account in Toma
            # if red_z_func == "distri":
            #     theta_j = acc_reject(func_theta_j) * ((1 + red_z_loop_func) / 2) ** -0.45
            # else:
            theta_j_loop_func = acc_reject(distrib_theta_j, [], 0.001, 0.2)
        else:
            theta_j_loop_func = theta_j_func

        if theta_nu_func == "distri_pearce":
            theta_nu_loop_func = generator_theta_nu(theta_j_loop_func, gamma_loop_func)
        elif theta_nu_func == "distri_toma":
            theta_nu_loop_func = acc_reject(distrib_theta_nu_toma, [], 0, 0.22)
        else:
            theta_nu_loop_func = theta_nu_func
            # theta_nu_loop_func = theta_nu_func * theta_j_loop_func

        if nu_0_func == "distri":
            nu_0_loop_func = generator_nu_0(theta_j_loop_func, gamma_loop_func)
        else:
            nu_0_loop_func = nu_0_func

        if alpha_func == "distri":
            alpha_loop_func = acc_reject(distrib_alpha, [], -1.6, 0.06)
        else:
            alpha_loop_func = alpha_func

        if beta_func == "distri":
            beta_loop_func = acc_reject(distrib_beta, [], -3.32, -1.6)
        else:
            beta_loop_func = beta_func

        alpha_loop_func = -(alpha_loop_func + 1)
        beta_loop_func = -(beta_loop_func + 1)
        # Calculation to determine whether of not the flux is supposed to be detected
        # luminosity distance has to be changed from Mpc to cm
        lum_dist = cosmology.luminosity_distance(red_z_loop_func).to_value("cm")  # Gpc
        # Core flux initiated with a luminosity of 10^52 erg/s (flux in erg/cm2/s)
        core_flux = 1e52 / (4 * np.pi * lum_dist**2)
        # print(core_flux)
        if flux_rejection:
            calculated_flux = jet_shape(theta_nu_loop_func, theta_j_loop_func, gamma_loop_func, jet_model, core_flux)
        else:
            calculated_flux = limit_flux
    #     if calculated_flux < limit_flux:
    #         print("q = ", theta_nu_loop_func/theta_j_loop_func)
    # print("q kept !!!!!! q = ", theta_nu_loop_func/theta_j_loop_func)
    return [iteration, gamma_loop_func, red_z_loop_func, theta_j_loop_func, theta_nu_loop_func, nu_0_loop_func,
            alpha_loop_func, beta_loop_func]


# def var_ite_setting_toma(iteration, gamma_func, red_z_func, theta_j_func, theta_nu_func, nu_0_func, alpha_func, beta_func):
#     """
#     Function to obtain a set of parameters according to simulation settings and distributions
#     This function is specific to the tests for Toma curves, so that que get fixed values for yj and a range for q
#     But as we force theta_j and theta_nu and not yj and q, some adjustments are needed (only on theta_nu_loop_func actually)
#     """
#     if gamma_func == "distri":
#         print("No distri added for gamma yet, value 100 is taken")
#         gamma_loop_func = 100
#     else:
#         gamma_loop_func = gamma_func
#
#     if red_z_func == "distri":
#         red_z_loop_func = acc_reject(distrib_z)
#     else:
#         red_z_loop_func = red_z_func
#
#     if theta_j_func == "distri":
#         theta_j_loop_func = acc_reject(distrib_theta_j)
#     else:
#         theta_j_loop_func = theta_j_func
#
#     if theta_nu_func == "distri_pearce":
#         theta_nu_loop_func = generator_theta_nu(theta_j_loop_func, gamma_loop_func)
#     elif theta_nu_func == "distri_toma":
#         theta_nu_loop_func = acc_reject(distrib_theta_nu_toma)
#     else:
#         theta_nu_loop_func = theta_nu_func * theta_j_loop_func
#
#     if nu_0_func == "distri":
#         nu_0_loop_func = generator_nu_0(theta_j_loop_func, gamma_loop_func)
#     else:
#         nu_0_loop_func = nu_0_func
#
#     if alpha_func == "distri":
#         alpha_loop_func = acc_reject(distrib_alpha)
#     else:
#         alpha_loop_func = alpha_func
#
#     if beta_func == "distri":
#         beta_loop_func = acc_reject(distrib_beta)
#     else:
#         beta_loop_func = beta_func
#
#     alpha_loop_func = -(alpha_loop_func + 1)
#     beta_loop_func = -(beta_loop_func + 1)
#     return [iteration, gamma_loop_func, red_z_loop_func, theta_j_loop_func, theta_nu_loop_func, nu_0_loop_func,
#             alpha_loop_func, beta_loop_func]
#

def jet_shape(theta_nu, theta_j, gamma, jet_structure, lum_flux_init):
    """
    Returns the luminosity or a flux at a given angle
    Formula from Pearce, but a - is mission in the article
    """
    if jet_structure == "top-hat":
        if theta_nu <= theta_j:
            return lum_flux_init
        else:
            return lum_flux_init * np.exp(-gamma**2 * (theta_nu - theta_j)**2 / 2)

########################################################################################################################
# Functions used to calculate integral from SO model                                                                   #
########################################################################################################################
def calc_x(z_func, nu_func, y_func, gamma_nu_0_func):
    return (1 + z_func) * nu_func * (1 + y_func) / (2 * gamma_nu_0_func)


def delta_phi(q_func, y_func, yj_func):
    if q_func > 1:
        val = np.where(y_func < (1 - q_func) ** 2 * yj_func, 1, ((q_func ** 2 - 1) * yj_func + y_func) / (2 * q_func * np.sqrt(yj_func * y_func)))
    elif q_func < 1:
        val = np.where(y_func < (1 - q_func) ** 2 * yj_func, -1, ((q_func ** 2 - 1) * yj_func + y_func) / (2 * q_func * np.sqrt(yj_func * y_func)))
    else:
        val = ((q_func ** 2 - 1) * yj_func + y_func) / (2 * q_func * np.sqrt(yj_func * y_func))
    return np.arccos(val)


def f_tilde(x_func, alpha_func, beta_func):
    if type(x_func) != np.ndarray:
        x_func = np.array(x_func)
    return np.where(x_func <= beta_func - alpha_func, x_func ** (-alpha_func) * np.exp(-x_func),
                    x_func ** (-beta_func) * (beta_func - alpha_func) ** (beta_func - alpha_func) * np.exp(alpha_func - beta_func))


def sin_theta_b(y_func, a_func, phi_func):
    return np.sqrt(((1 - y_func) / (1 + y_func)) ** 2 + 4 * y_func / (1 + y_func) ** 2 * (a_func - np.cos(phi_func)) ** 2 /
                   (1 + a_func ** 2 - 2 * a_func * np.cos(phi_func)))


# def sin_theta_b_opti(y_func, a_func, phi_func):
#     if type(y_func) != np.ndarray:
#         y_func = np.array(y_func)
#     if type(a_func) != np.ndarray:
#         a_func = np.array(a_func)
#     y_func = np.reshape(y_func, (len(y_func), 1))
#     a_func = np.reshape(a_func, (len(a_func), 1))
#     return np.sqrt(
#         ((1 - y_func) / (1 + y_func)) ** 2 + 4 * y_func / (1 + y_func) ** 2 * (a_func - np.cos(phi_func)) ** 2 / (
#                 1 + a_func ** 2 - 2 * a_func * np.cos(phi_func)))


def pi_syn(x_func, alpha_func, beta_func):
    if type(x_func) != np.ndarray:
        x_func = np.array(x_func)
    return np.where(x_func <= beta_func - alpha_func, (alpha_func + 1) / (alpha_func + 5 / 3), (beta_func + 1) / (beta_func + 5 / 3))


def ksi(y_func, a_func, phi_func):
    return phi_func + np.arctan((1 - y_func) / (1 + y_func) * np.sin(phi_func) / (a_func - np.cos(phi_func)))

# def ksi2(y_func, a_func, phi_func):
#     if type(y_func) != np.ndarray:
#         y_func = np.array(y_func)
#     if type(a_func) != np.ndarray:
#         a_func = np.array(a_func)
#     y_func = np.reshape(y_func, (len(y_func), 1))
#     a_func = np.reshape(a_func, (len(a_func), 1))
#     return phi_func + np.arctan((1 - y_func) / (1 + y_func) * np.sin(phi_func) / (a_func - np.cos(phi_func)))


# def ksi_opti(y_func, a_func, phi_func):
#     if type(y_func) != np.ndarray:
#         y_func = np.array(y_func)
#     if type(a_func) != np.ndarray:
#         a_func = np.array(a_func)
#     y_func = np.reshape(y_func, (len(y_func), 1))
#     a_func = np.reshape(a_func, (len(a_func), 1))
#     return phi_func + np.arctan((1 - y_func) / (1 + y_func) * np.sin(phi_func) / (a_func - np.cos(phi_func)))


def val_moy_sin_cos(eta_func, y_func, alpha_func):
    return (1 - 4 * y_func / (1 + y_func)**2 * (np.cos(eta_func))**2)**((alpha_func - 1) / 2) * \
        ((np.sin(eta_func))**2 - ((1 - y_func) / (1 + y_func))**2 * (np.cos(eta_func))**2)


def val_moy_sin(eta_func, y_func, alpha_func):
    return (1 - 4 * y_func / (1 + y_func)**2 * (np.cos(eta_func))**2)**((alpha_func + 1) / 2)


def error_calc(num, num_std, denom, denom_std, iteration_number, confidence=1.96):
    """
    Function to calculate the error of a value that has the shape value = num/denom
    Knowing num, num_std, denom, denom_std
    """
    num = np.abs(num)
    # std = num / denom * np.sqrt((num_std / num)**2 + (denom_std / denom)**2)
    std = np.sqrt((num_std/denom)**2 + (num * denom_std / denom**2)**2)
    return std / np.sqrt(iteration_number) * confidence


def use_scipyquad(func, low_edge, high_edge, func_args=(), x_logscale=False):
  """
  Proceed to the quad integration using scipy quad and handle the integration warning by using simpson integration method from scipy if the warning is raised
  """
  if type(func_args) != tuple:
    raise TypeError(f"Function use_scipyquad takes func_args as tuple only, {type(func_args)} given")
  try:
    warnings.simplefilter("error", category=IntegrationWarning)
    int_spectrum, err = quad(func, low_edge, high_edge, args=func_args)
    return int_spectrum, err
  except IntegrationWarning:
    if x_logscale:
      int_x = np.logspace(np.log10(low_edge), np.log10(high_edge), 10000001)
    else:
      int_x = np.linspace(low_edge, high_edge, 10000001)
    int_y = func(int_x, *func_args)
    int_spectrum = simpson(int_y, x=int_x)
    return int_spectrum, None
