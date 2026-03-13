# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  models.py
# Contains functions to computes the polarisaiton fraction according to different GRB jet models
# ================================================================

# Regular imports
import numpy as np
import os
import time

# Developped modules imports
from src.General.funcmod import calc_x, delta_phi, f_tilde, sin_theta_b, pi_syn, ksi, val_moy_sin_cos, val_moy_sin, error_calc

seed1 = 1
seed2 = 2
seed3 = 3


def integral_calculation_SO(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Computes the polarization fraction for the Synchrotron Ordered (SO) magnetic field model using a Monte Carlo triple integral.
    Integration is vectorised by reshaping the random sample arrays into broadcastable tensors:
        nu_integ_list  -> shape (N, 1, 1)
        y_integ_list   -> shape (N, N, 1)
        phi_integ_list -> shape (N, N, N)
    so that numpy broadcasts all three into a final (N, N, N) tensor that is summed in one pass.
    :param nu_min: float, lower bound of the observed frequency integration range [keV]
    :param nu_max: float, upper bound of the observed frequency integration range [keV]
    :param gamma: float, Lorentz factor of the jet
    :param red_z: float, source redshift
    :param theta_j: float, jet half-opening angle [rad]
    :param theta_nu: float, off-axis viewing angle [rad]
    :param nu_0: float, reference frequency used to compute gamma_nu_0 [keV]
    :param alpha: float, low-energy Band spectral index
    :param beta: float, high-energy Band spectral index
    :param integ_steps: int, number of Monte Carlo sample points per integration dimension; total sample count is integ_steps**3, default=70
    :param confidence: float, z-score multiplier used by error_calc to compute the confidence interval on the polarization fraction, default=1.96
    :param seed: bool, if True fixes the RNG seed to 1 for reproducibility
        if False seeds from PID and wall-clock time, default=False
    :returns: float, float, np.ndarray, the polarization fraction, its
        confidence-interval half-width, and the raw numerator integrand array
        num of shape (N, N, N) (used for diagnostics in the caller)
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)

    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0

    # Random values :
    rand_nu = np.random.random(integ_steps).reshape((integ_steps, 1, 1))
    rand_y = np.random.random((integ_steps, integ_steps)).reshape((integ_steps, integ_steps, 1))
    rand_phi = np.random.random((integ_steps, integ_steps, integ_steps))

    # Integration ranges and integration random points
    rand_range_1 = (nu_max - nu_min)
    rand_range_2 = (1 + q_var) ** 2 * yj

    nu_integ_list = nu_min + rand_nu * rand_range_1
    y_integ_list = rand_y * rand_range_2

    # Setting intermediate variables using parameters and integration points
    a_var = np.sqrt(y_integ_list / yj) / q_var
    x_var = calc_x(red_z, nu_integ_list, y_integ_list, gamma_nu_0)
    delta_phi_val = delta_phi(q_var, y_integ_list, yj)

    # Integration range and integration random points
    rand_range_3 = 2 * delta_phi_val
    phi_integ_list = -delta_phi_val + rand_phi * rand_range_3

    # Setting intermediate functions using parameters and integration points
    f_tilde_val = f_tilde(x_var, alpha, beta)
    sin_theta_b_val = sin_theta_b(y_integ_list, a_var, phi_integ_list)
    pi_syn_val = pi_syn(x_var, alpha, beta)
    ksi_val = ksi(y_integ_list, a_var, phi_integ_list)

    norm = rand_range_2 * rand_range_1 / integ_steps**3

    num = f_tilde_val * sin_theta_b_val ** (alpha + 1) * pi_syn_val * np.cos(2 * ksi_val) / (1 + y_integ_list) ** 2 * rand_range_3
    denom = f_tilde_val * sin_theta_b_val ** (alpha + 1) / (1 + y_integ_list) ** 2 * rand_range_3
    numsum = np.sum(num) * norm
    denomsum = np.sum(denom) * norm

    num_sqared_sum = np.sum(np.power(num, 2)) * norm ** 2
    denom_sqared_sum = np.sum(np.power(denom, 2)) * norm**2
    std_num = np.sqrt(np.abs(num_sqared_sum - numsum ** 2))
    std_denom = np.sqrt(np.abs(denom_sqared_sum - denomsum**2))

    return abs(numsum) / denomsum, error_calc(numsum, std_num, denomsum, std_denom, integ_steps**3, confidence=confidence), num


def integral_calculation_SR(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Computes the polarization fraction for the Synchrotron Random (SR) magnetic field model using a Monte Carlo triple integral.
    Integration is vectorised by reshaping the random sample arrays into broadcastable tensors:
        nu_integ_list  -> shape (N, 1, 1)
        y_integ_list   -> shape (N, N, 1)
        eta_integ_list -> shape (N, N, N)
    so that numpy broadcasts all three into a final (N, N, N) tensor that is summed in one pass.
    :param nu_min: float, lower bound of the observed frequency integration range [keV]
    :param nu_max: float, upper bound of the observed frequency integration range [keV]
    :param gamma: float, Lorentz factor of the jet
    :param red_z: float, source redshift
    :param theta_j: float, jet half-opening angle [rad]
    :param theta_nu: float, off-axis viewing angle [rad]
    :param nu_0: float, reference frequency used to compute gamma_nu_0 [keV]
    :param alpha: float, low-energy Band spectral index
    :param beta: float, high-energy Band spectral index
    :param integ_steps: int, number of Monte Carlo sample points per integration dimension; total sample count is integ_steps**3, default=70
    :param confidence: float, z-score multiplier used by error_calc to compute the confidence interval on the polarization fraction, default=1.96
    :param seed: bool, if True fixes the RNG seed to 1 for reproducibility
        if False seeds from PID and wall-clock time, default=False
    :returns: float, float, the polarization fraction and its confidence-interval half-width
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)

    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0

    # Integration ranges and integration random points
    rand_range_1 = (nu_max - nu_min)
    rand_range_2 = (1 + q_var) ** 2 * yj
    rand_range_3 = np.pi

    # Random values
    rand_nu = np.random.random(integ_steps).reshape((integ_steps, 1, 1))
    rand_y = np.random.random((integ_steps, integ_steps)).reshape((integ_steps, integ_steps, 1))
    rand_phi = np.random.random((integ_steps, integ_steps, integ_steps))

    nu_integ_list = nu_min + rand_nu * rand_range_1
    y_integ_list = rand_y * rand_range_2
    eta_integ_list = rand_phi * rand_range_3

    # Setting intermediate variables using parameters and integration points
    x_var = calc_x(red_z, nu_integ_list, y_integ_list, gamma_nu_0)
    delta_phi_val = delta_phi(q_var, y_integ_list, yj)
    f_tilde_val = f_tilde(x_var, alpha, beta)
    pi_syn_val = pi_syn(x_var, alpha, beta)

    norm = rand_range_3 * rand_range_2 * rand_range_1 / integ_steps**3

    # Calculating the value in the integral
    num = f_tilde_val * pi_syn_val * val_moy_sin_cos(eta_integ_list, y_integ_list, alpha) * np.sin(2 * delta_phi_val) / (1 + y_integ_list) ** 2
    denom = f_tilde_val * val_moy_sin(eta_integ_list, y_integ_list, alpha) * 2 * delta_phi_val / (1 + y_integ_list) ** 2
    numsum = np.sum(num) * norm
    denomsum = np.sum(denom) * norm

    num_sqared_sum = np.sum(np.power(num, 2)) * norm ** 2
    denom_sqared_sum = np.sum(np.power(denom, 2)) * norm ** 2
    std_num = np.sqrt(np.abs(num_sqared_sum - numsum ** 2))
    std_denom = np.sqrt(np.abs(denom_sqared_sum - denomsum**2))

    return abs(numsum) / denomsum, error_calc(numsum, std_num, denomsum, std_denom, integ_steps**3, confidence=confidence)


def integral_calculation_CD(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Computes the polarization fraction for the Compton Drag (CD) model using a Monte Carlo double integral.
    Integration is vectorised by reshaping the random sample arrays into broadcastable tensors:
        nu_integ_list -> shape (N, 1, 1)
        y_integ_list  -> shape (N, N, 1)
    so that numpy broadcasts both into a final (N, N) tensor that is summed in one pass.
    :param nu_min: float, lower bound of the observed frequency integration range [keV]
    :param nu_max: float, upper bound of the observed frequency integration range [keV]
    :param gamma: float, Lorentz factor of the jet
    :param red_z: float, source redshift
    :param theta_j: float, jet half-opening angle [rad]
    :param theta_nu: float, off-axis viewing angle [rad]
    :param nu_0: float, reference frequency used to compute gamma_nu_0 [keV]
    :param alpha: float, low-energy Band spectral index
    :param beta: float, high-energy Band spectral index
    :param integ_steps: int, number of Monte Carlo sample points per integration dimension; total sample count is integ_steps**2, default=70
    :param confidence: float, z-score multiplier used by error_calc to compute the confidence interval on the polarization fraction, default=1.96
    :param seed: bool, if True fixes the RNG seed to 1 for reproducibility
        if False seeds from PID and wall-clock time, default=False
    :returns: float, float, the polarization fraction and its confidence-interval half-width
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)

    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0

    # Integration ranges and integration random points
    rand_range_1 = (nu_max - nu_min)
    rand_range_2 = (1 + q_var) ** 2 * yj

    # Random values
    rand_nu = np.random.random(integ_steps).reshape((integ_steps, 1, 1))
    rand_y = np.random.random((integ_steps, integ_steps)).reshape((integ_steps, integ_steps, 1))
    nu_integ_list = nu_min + rand_nu * rand_range_1
    y_integ_list = rand_y * rand_range_2

    # Setting intermediate variables using parameters and integration points
    x_var = calc_x(red_z, nu_integ_list, y_integ_list, gamma_nu_0)
    delta_phi_val = delta_phi(q_var, y_integ_list, yj)
    f_tilde_val = f_tilde(x_var, alpha, beta)

    norm = rand_range_2 * rand_range_1 / integ_steps**2

    num = f_tilde_val * 2 * y_integ_list / (1 + y_integ_list) ** 4 * np.sin(2 * delta_phi_val)
    denom = f_tilde_val * (1 + y_integ_list ** 2) / (1 + y_integ_list) ** 4 * 2 * delta_phi_val

    numsum = np.sum(num) * norm
    denomsum = np.sum(denom) * norm

    num_sqared_sum = np.sum(np.power(num, 2)) * norm ** 2
    denom_sqared_sum = np.sum(np.power(denom, 2)) * norm ** 2
    std_num = np.sqrt(np.abs(num_sqared_sum - numsum ** 2))
    std_denom = np.sqrt(np.abs(denom_sqared_sum - denomsum**2))

    return abs(numsum) / denomsum, error_calc(numsum, std_num, denomsum, std_denom, integ_steps**3, confidence=confidence)


def integral_calculation_PJ(gamma, theta_j, theta_nu):
    """
    Computes the polarization fraction for the Patchy Jet (PJ) model using an analytic Gaussian profile centred on the jet edge.
    :param gamma: float, Lorentz factor of the jet
    :param theta_j: float, jet half-opening angle [rad]
    :param theta_nu: float, off-axis viewing angle [rad]
    :returns: float, float, the polarization fraction and the Gaussian width sigma_pj = 1/gamma [rad]
    """
    pf_max = 0.4
    sigma_pj = 1 / gamma
    return pf_max * np.exp(-(theta_nu - theta_j) ** 2 / (2 * sigma_pj ** 2)), sigma_pj
