# Regular imports
import numpy as np
import os
import time

# Developped modules imports
from src.General.funcmod import calc_x, delta_phi, f_tilde, sin_theta_b, pi_syn, ksi, val_moy_sin_cos, val_moy_sin, error_calc

seed1 = 1
seed2 = 2
seed3 = 3

def integral_calculation_SO_old(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Optimized function to calculate the polarization fraction for the SO model
    Optimization has been done reshaping the integration point lists :
        nu_integ_list to a tensor (N, 1, 1) : np.reshape(vec, (N, 1, 1))
        y_integ_list to a tensor (N, 1) : np.reshape(vec, (N, 1, 1))
        phi_integ_list to a tensor (N) : no change to this vector
    Then the way numpy handles the operation between tensors makes a final tensor of size (N, N, N) that can then be summed
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)
    # Step 2 : The integration starts here, definition of 2 first integration paramaters ==> 2 loops
    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0
    ################################################################################################################################
    # Calculation of the 1st integral
    ################################################################################################################################
    # Integration range and integration random points
    rand_range_1 = (nu_max - nu_min)
    # nu_integ_list = nu_min + np.reshape(np.random.random(integ_steps) * rand_range_1, (integ_steps, 1, 1))
    nu_integ_list = nu_min + np.random.random(integ_steps) * rand_range_1

    ################################################################################################################################
    # Calculation of the 2nd integral
    ################################################################################################################################
    # Integration range and integration random points
    rand_range_2 = (1 + q_var) ** 2 * yj
    y_integ_list = np.random.random((integ_steps, integ_steps)) * rand_range_2
    # Setting intermediate variables using parameters and integration points
    a_var = np.sqrt(y_integ_list / yj) / q_var
    x_var = calc_x(red_z, nu_integ_list, y_integ_list, gamma_nu_0)
    delta_phi_val = delta_phi(q_var, y_integ_list, yj)

    ################################################################################################################################
    # Calculation of the 3rd integral with numpy arrays
    ################################################################################################################################
    # Integration range and integration random points
    rand_range_3 = 2 * delta_phi_val
    phi_integ_list = -delta_phi_val + np.random.random((integ_steps, integ_steps, integ_steps)) * rand_range_3
    # Setting intermediate functions using parameters and integration points
    f_tilde_val = f_tilde(x_var, alpha, beta)
    sin_theta_b_val = sin_theta_b(y_integ_list, a_var, phi_integ_list)
    pi_syn_val = pi_syn(x_var, alpha, beta)
    ksi_val = ksi(y_integ_list, a_var, phi_integ_list)

    norm_moy = rand_range_3 * rand_range_2 * rand_range_1 / integ_steps**3
    norm_var = (rand_range_3 * rand_range_2 * rand_range_1)**2 / integ_steps**3
    # Calculating the value in the integral
    num_integ = f_tilde_val * sin_theta_b_val ** (alpha + 1) * pi_syn_val * np.cos(2 * ksi_val) / (1 + y_integ_list) ** 2
    denom_integ = f_tilde_val * sin_theta_b_val ** (alpha + 1) / (1 + y_integ_list) ** 2

    std_num = num_integ * num_integ
    std_denom = denom_integ * denom_integ

    # Calculating the 1st, 2nd and 3rd integral by summing along axis
    num_integ = np.sum(num_integ * norm_moy)
    denom_integ = np.sum(denom_integ * norm_moy)

    std_num = np.sqrt((np.sum(std_num * norm_var) - num_integ**2))
    std_denom = np.sqrt((np.sum(std_denom * norm_var) - denom_integ**2))
    # Step 7 : Obtention of PF
    return abs(num_integ) / denom_integ, error_calc(num_integ, std_num, denom_integ, std_denom, integ_steps**3, confidence=confidence)


def integral_calculation_SO(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Optimized function to calculate the polarization fraction for the SO model
    Optimization has been done reshaping the integration point lists :
        nu_integ_list to a tensor (N, 1, 1) : np.reshape(vec, (N, 1, 1))
        y_integ_list to a tensor (N, 1) : np.reshape(vec, (N, 1, 1))
        phi_integ_list to a tensor (N) : no change to this vector
    Then the way numpy handles the operation between tensors makes a final tensor of size (N, N, N) that can then be summed
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)

    # Step 2 : The integration starts here, definition of 2 first integration paramaters ==> 2 loops
    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0

    # Random values :
    rand_nu = np.random.random(integ_steps).reshape((integ_steps, 1, 1))
    rand_y = np.random.random((integ_steps, integ_steps)).reshape((integ_steps, integ_steps, 1))
    rand_phi = np.random.random((integ_steps, integ_steps, integ_steps))
    ################################################################################################################################
    # Calculation of the 1st integral
    ################################################################################################################################
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
    # if len(vals1) == 0:
    #     # print("delta phi : ", delta_phi_val)
    #     # print("a : ", a_var)
    #     # print("y list : ", y_integ_list)
    #     # print("phi : ", phi_integ_list)
    #     print("f_tilde : ", f_tilde_val)
    #     print("sin : ", sin_theta_b_val)
    #     print("pi : ", pi_syn_val)
    #     print("ksi : ", ksi_val)
    # print("random phi", rand_phi)
    # print("delta phi", delta_phi_val)
    # print("nu_integ_list", nu_integ_list)
    # print("y_integ_list", y_integ_list)
    # print("phi_integ_list", phi_integ_list)

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


def integral_calculation_SO3(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Optimized function to calculate the polarization fraction for the SO model
    Optimization has been done reshaping the integration point lists :
        nu_integ_list to a tensor (N, 1, 1) : np.reshape(vec, (N, 1, 1))
        y_integ_list to a tensor (N, 1) : np.reshape(vec, (N, 1, 1))
        phi_integ_list to a tensor (N) : no change to this vector
    Then the way numpy handles the operation between tensors makes a final tensor of size (N, N, N) that can then be summed
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)

    # Step 2 : The integration starts here, definition of 2 first integration paramaters ==> 2 loops
    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0

    # Random values
    rand_nu = np.random.random(integ_steps)
    rand_y = np.random.random((integ_steps, integ_steps))
    rand_phi = np.random.random((integ_steps, integ_steps, integ_steps))

    ################################################################################################################################
    # Calculation of the 1st integral
    ################################################################################################################################
    # Integration range and integration random points
    rand_range_1 = (nu_max - nu_min)
    # np.random.seed(seed1)
    nu_integ_list = nu_min + rand_nu * rand_range_1

    num = 0
    denom = 0
    vals1 = []
    num1 = 0
    denom1 = 0
    for nuite, nuval in enumerate(nu_integ_list):
        ################################################################################################################################
        # Calculation of the 2nd integral
        ################################################################################################################################
        # Integration range and integration random points
        rand_range_2 = (1 + q_var) ** 2 * yj
        # np.random.seed(seed2)
        y_integ_list = rand_y[nuite] * rand_range_2
        vals2 = []
        num2 = 0
        denom2 = 0
        for yite, yval in enumerate(y_integ_list):
            # Setting intermediate variables using parameters and integration points
            a_var = np.sqrt(yval / yj) / q_var
            x_var = calc_x(red_z, nuval, yval, gamma_nu_0)
            delta_phi_val = delta_phi(q_var, yval, yj)

            ################################################################################################################################
            # Calculation of the 3rd integral with numpy arrays
            ################################################################################################################################
            # Integration range and integration random points
            rand_range_3 = 2 * delta_phi_val
            # np.random.seed(seed3)
            phi_integ_list = -delta_phi_val + rand_phi[nuite][yite] * rand_range_3
            vals3 = []
            num3 = 0
            denom3 = 0
            for phival in phi_integ_list:
                # Setting intermediate functions using parameters and integration points
                f_tilde_val = f_tilde(x_var, alpha, beta)
                sin_theta_b_val = sin_theta_b(yval, a_var, phival)
                pi_syn_val = pi_syn(x_var, alpha, beta)
                ksi_val = ksi(yval, a_var, phival)
                # if len(vals1) == 0:
                # print("REF random phi : ", rand_phi)
                # print("REF delta phi", delta_phi_val)
                # print("REF nu_integ_list", nu_integ_list)
                # print("REF y_integ_list", y_integ_list)
                # print("REF phi_integ_list", phi_integ_list)

                #     # print("REF x : ", x_var)
                #     # print("REF y : ", yval)
                #     # print("REF a : ", a_var)
                #     # print("REF phi : ", a_var)
                #     # print("REF delta phi : ", delta_phi_val)
                #     print("REF sin : ", sin_theta_b_val)
                #     print("REF ksi : ", ksi_val)

                valnum = f_tilde_val * sin_theta_b_val ** (alpha + 1) * pi_syn_val * np.cos(2 * ksi_val) / (1 + yval) ** 2 * rand_range_3
                num3 += valnum
                denom3 += f_tilde_val * sin_theta_b_val ** (alpha + 1) / (1 + yval) ** 2 * rand_range_3
                vals3.append(valnum)

            num2 += num3 / integ_steps
            denom2 += denom3 / integ_steps
            vals2.append(vals3)

        num1 += num2 * rand_range_2 / integ_steps
        denom1 += denom2 * rand_range_2 / integ_steps
        vals1.append(vals2)

    num = num1 * rand_range_1 / integ_steps
    denom = denom1 * rand_range_1 / integ_steps
    return abs(num) / denom, 0, vals1


def integral_calculation_SR_old(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Optimized function to calculate the polarization fraction for the SR model
    Optimization has been done reshaping the integration point lists :
        nu_integ_list to a tensor (N, 1, 1) : np.reshape(vec, (N, 1, 1))
        y_integ_list to a tensor (N, 1) : np.reshape(vec, (N, 1, 1))
        eta_integ_list to a tensor (N) : no change to this vector
    Then the way numpy handles the operation between tensors makes a final tensor of size (N, N, N) that can then be summed
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)

    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0
    ################################################################################################################################
    # Calculation of the 1st integral
    ################################################################################################################################
    # Integration range and integration random points
    rand_range_1 = (nu_max - nu_min)
    # np.random.seed(seed1)
    nu_integ_list = nu_min + np.random.random(integ_steps) * rand_range_1

    ################################################################################################################################
    # Calculation of the 2nd integral
    ################################################################################################################################
    # Integration range and integration random points
    rand_range_2 = (1 + q_var) ** 2 * yj
    # np.random.seed(seed2)
    y_integ_list = np.random.random((integ_steps, integ_steps)) * rand_range_2
    # Setting intermediate variables using parameters and integration points
    x_var = calc_x(red_z, nu_integ_list, y_integ_list, gamma_nu_0)
    delta_phi_val = delta_phi(q_var, y_integ_list, yj)
    f_tilde_val = f_tilde(x_var, alpha, beta)
    pi_syn_val = pi_syn(x_var, alpha, beta)

    ################################################################################################################################
    # Calculation of the 3rd integral with numpy arrays
    ################################################################################################################################
    # Integration range and integration random points
    rand_range_3 = np.pi
    # np.random.seed(seed3)
    eta_integ_list = np.random.random((integ_steps, integ_steps, integ_steps)) * rand_range_3
    # In the following lines we don't multiply by rand_range_3 as we have rand_range_3 * np.pi = 1 (due to formula)
    norm_moy = rand_range_2 * rand_range_1 / integ_steps**3
    norm_var = (rand_range_2 * rand_range_1)**2 / integ_steps**3

    # Calculating the value in the integral
    num_integ = f_tilde_val * pi_syn_val * val_moy_sin_cos(eta_integ_list, y_integ_list, alpha) * np.sin(2 * delta_phi_val) / (1 + y_integ_list) ** 2
    denom_integ = f_tilde_val * val_moy_sin(eta_integ_list, y_integ_list, alpha) * 2 * delta_phi_val / (1 + y_integ_list) ** 2

    std_num = num_integ * num_integ
    std_denom = denom_integ * denom_integ

    # Calculating the 1st, 2nd and 3rd integral by summing along axis
    num_integ = np.sum(num_integ) * norm_moy
    denom_integ = np.sum(denom_integ) * norm_moy

    std_num = np.sqrt((np.sum(std_num * norm_var) - num_integ**2))
    std_denom = np.sqrt((np.sum(std_denom * norm_var) - denom_integ**2))
    # Step 7 : Obtention of PF
    return abs(num_integ) / denom_integ, error_calc(num_integ, std_num, denom_integ, std_denom, integ_steps**3, confidence=confidence)


def integral_calculation_SR(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Optimized function to calculate the polarization fraction for the SR model
    Optimization has been done reshaping the integration point lists :
        nu_integ_list to a tensor (N, 1, 1) : np.reshape(vec, (N, 1, 1))
        y_integ_list to a tensor (N, 1) : np.reshape(vec, (N, 1, 1))
        eta_integ_list to a tensor (N) : no change to this vector
    Then the way numpy handles the operation between tensors makes a final tensor of size (N, N, N) that can then be summed
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
    # rand_nu = np.random.random(integ_steps)
    # rand_y = np.random.random((integ_steps, integ_steps))
    # rand_phi = np.random.random((integ_steps, integ_steps, integ_steps))
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

def integral_calculation_CD_old(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Optimized function to calculate the polarization fraction for the CD model
    Optimization has been done reshaping the integration point lists :
        nu_integ_list to a tensor (N, 1) : np.reshape(vec, (N, 1, 1))
        y_integ_list to a tensor (N) : no change to this vector
    Then the way numpy handles the operation between tensors makes a final tensor of size (N, N, N) that can then be summed
    """
    #initiating a random seed:
    if seed:
        np.random.seed(1)
    else:
        np.random.seed((os.getpid() + int(time.time() * 1000)) % 2 ** 32)

    q_var = theta_nu / theta_j
    yj = (theta_j * gamma)**2
    gamma_nu_0 = gamma * nu_0

    # Integration list for the 1st integral
    rand_range_1 = (nu_max - nu_min)
    # np.random.seed(seed1)
    nu_integ_list = nu_min + np.random.random(integ_steps) * rand_range_1

    # Integration list for the 2nd integral
    rand_range_2 = (1 + q_var) ** 2 * yj
    # np.random.seed(seed2)
    y_integ_list = np.random.random((integ_steps, integ_steps)) * rand_range_2

    # Calculation of the 2nd integral
    x_var = calc_x(red_z, nu_integ_list, y_integ_list, gamma_nu_0)
    delta_phi_val = delta_phi(q_var, y_integ_list, yj)
    f_tilde_val = f_tilde(x_var, alpha, beta)

    norm_moy = rand_range_2 * rand_range_1 / integ_steps**2
    norm_var = (rand_range_2 * rand_range_1)**2 / integ_steps**2

    # Calculating the value in the integral
    num_integ = f_tilde_val * 2 * y_integ_list / (1 + y_integ_list) ** 2 * np.sin(2 * delta_phi_val) / (1 + y_integ_list) ** 2
    denom_integ = f_tilde_val * (1 + y_integ_list ** 2) / (1 + y_integ_list) ** 2 * 2 * delta_phi_val / (1 + y_integ_list) ** 2

    std_num = num_integ * num_integ
    std_denom = denom_integ * denom_integ

    # Calculating the 1st, 2nd and 3rd integral by summing along axis
    num_integ = np.sum(num_integ) * norm_moy
    denom_integ = np.sum(denom_integ) * norm_moy

    std_num = np.sqrt((np.sum(std_num * norm_var) - num_integ**2))
    std_denom = np.sqrt((np.sum(std_denom * norm_var) - denom_integ**2))
    # Step 7 : Obtention of PF
    return abs(num_integ) / denom_integ, error_calc(num_integ, std_num, denom_integ, std_denom, integ_steps**2, confidence=confidence)


def integral_calculation_CD(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96, seed=False):
    """
    Optimized function to calculate the polarization fraction for the CD model
    Optimization has been done reshaping the integration point lists :
        nu_integ_list to a tensor (N, 1) : np.reshape(vec, (N, 1, 1))
        y_integ_list to a tensor (N) : no change to this vector
    Then the way numpy handles the operation between tensors makes a final tensor of size (N, N, N) that can then be summed
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
    Function to calculate the polarization fraction for the PJ model
    """
    pf_max = 0.4
    sigma_pj = 1 / gamma
    return pf_max * np.exp(-(theta_nu - theta_j) ** 2 / (2 * sigma_pj ** 2)), sigma_pj
