import numpy as np


# ########################################################################################################################
# # Acceptance rejection method                                                                                          #
# ########################################################################################################################
# def acc_reject(func, *func_args):
#     loop = True
#     while loop:
#         variable, thresh_value, max_func = func(*func_args)
#         test_value = np.random.random() * max_func
#         if test_value <= thresh_value:
#             # loop = False
#             return variable
#
#
# ########################################################################################################################
# # Functions used with acceptance rejection method to get values according distributions                                #
# ########################################################################################################################
# def distrib_alpha():
#     """
#     Alpha follows a distribution obtained from the GBM data, for GRB with best fit being band spectrum
#     """
#     histo = np.array([0.00465116, 0.00465116, 0.00930233, 0.01395349, 0.05581395, 0.08372093, 0.09302326, 0.10232558,
#                       0.10697674, 0.17674419, 0.11162791, 0.05581395, 0.06046512, 0.04651163, 0.02790698, 0.00930233,
#                       0.00930233, 0., 0.00930233, 0.01860465])
#     bins = np.array([-1.584363, -1.50239975, -1.4204365, -1.33847325, -1.25651, -1.17454676, -1.09258351, -1.01062026,
#                      -0.92865701, -0.84669376, -0.76473051, -0.68276726, -0.60080401, -0.51884076, -0.43687751,
#                      -0.35491427, -0.27295102, -0.19098777, -0.10902452, -0.02706127, 0.05490198])
#     x_rand = bins[0] + np.random.random() * (bins[-1] - bins[0])
#     index_bin = int((x_rand - bins[0]) / (bins[1] - bins[0]))
#     max_func = 0.17674419
#     return x_rand, histo[index_bin], max_func
#
#
# def distrib_beta():
#     """
#     Beta follows a distribution obtained from the GBM data, for GRB with best fit being band spectrum
#     """
#     histo = np.array([0.01860465, 0.00465116, 0.00465116, 0.01395349, 0.01860465, 0.01395349, 0.02325581, 0.02790698,
#                       0.04651163, 0.06046512, 0.06511628, 0.09302326, 0.13023256, 0.10697674, 0.12093023, 0.09302326,
#                       0.05116279, 0.01860465, 0.04651163, 0.04186047])
#     bins = np.array([-3.311648, -3.22707625, -3.1425045, -3.05793275, -2.973361, -2.88878925, -2.8042175,
#                      -2.71964575, -2.635074, -2.55050225, -2.4659305, -2.38135875, -2.296787, -2.21221525,
#                      -2.1276435, -2.04307175, -1.9585, -1.87392825, -1.7893565, -1.70478475, -1.620213])
#     x_rand = bins[0] + np.random.random() * (bins[-1] - bins[0])
#     index_bin = int((x_rand - bins[0]) / (bins[1] - bins[0]))
#     max_func = 0.13023256
#     return x_rand, histo[index_bin], max_func
#
#
# def distrib_theta_nu_toma():
#     """
#     theta nu follows a distribution given by Toma_2009
#     """
#     x_rand = np.random.random() * 0.22
#     max_func = np.sin(0.22)
#     return x_rand, np.sin(x_rand), max_func
#
#
# def distrib_theta_nu_2(thetaj):
#     """
#     theta nu follows a distribution given by Toma_2009 but it depends on the value of thetaj
#     Equivalent to the following function to get theta_nu from a uniform distribution
#     """
#     x_rand = np.random.random() * (thetaj + 5 / 100)
#     max_func = np.sin(0.02 + 5 / 100)
#     return x_rand, np.sin(x_rand), max_func
#
#
# def distrib_theta_j():
#     """
#     Distri theta j given by Toma_2009
#     q2 comes from observation of jet breaks and from analysis of BATSE, q1 highly uncertain
#     """
#     xmin = 0.001
#     xmax = 0.2
#     x_rand = xmin + np.random.random() * (xmax - xmin)
#     coupure = 0.02
#     max_func = 1
#     if x_rand <= coupure:
#         return x_rand, coupure**(-0.5) * x_rand**0.5, max_func
#     else:
#         return x_rand, coupure**2 * x_rand**(-2), max_func
#
#
# def distrib_z():
#     """
#     Distributions took considering that GRB rate is proportionnal to SFR rate
#     So distribution is proportionnal to SFR, which is function of z (giving the distribution)
#     Equation used given by Toma_2009 but seems to have different equations possible, might be interesting to search for
#     more recent ones (the equation used comes from Porciani_2001
#     zmax = 5, value taken from Dainotti_2023, may be a little high considering the shape of the distribution (doesn't
#     seem to be that much GRB at high z, but maybe selection effect of the platinum sample from Dainotti)
#     """
#     x_rand = np.random.random() * 5
#     max_func = 0.57
#     rate = np.exp(3.4 * x_rand) / (np.exp(3.4 * x_rand) + 22) * np.sqrt(0.3 * (1 + x_rand) ** 3 + 0.7) / (1 + x_rand) ** (3 / 2)
#     return x_rand, rate, max_func
#
#
# def generator_theta_nu(theta_j, gamma):
#     """
#     Generate a value for theta_nu using the transformation method
#     Values follows a distribution with a sin shape between theta_nu = 0 and theta_j + X/gamma value of X isn't clear
#     """
#     opening_factor = 5
#     return np.arccos(np.cos(theta_j + opening_factor / gamma) + np.random.random() * (1 - np.cos(theta_j + opening_factor / gamma)))
#
#
# def generator_nu_0(theta_j, gamma):
#     """
#     Generate a value for nu_0 according to the formula from Toma 2009
#     """
#     return 80 / gamma * np.random.lognormal(1, np.sqrt(0.15)) * np.sqrt(np.random.lognormal(1, np.sqrt(0.3)) / (5 * theta_j ** 2))
