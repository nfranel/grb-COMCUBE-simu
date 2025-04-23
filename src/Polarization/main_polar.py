# Regular imports
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Developped modules imports
from src.Polarization.polarization_class import PolVSAngleRatio
from src.Polarization.models import integral_calculation_SO

plt.rcParams.update({'font.size': 12})

########################################################################################################################
# test main distri                                                                                                     #
########################################################################################################################
activate_distrib = True
if activate_distrib:
    print("======================================================================================")
    print("Paramètres de simulation de la distribution")
    print("======================================================================================")
    int_step = 150
    jet_model_used = "top-hat"
    reject_low_flux = True
    confidence_option = 1.96 # no(not good), 8(better), 7(not bad), 6(best ?)
    print(f"Number of iterations for calculating the integrals : int_step = {int_step}")
    print(f"Model used for the jet : {jet_model_used}")
    print(f"Rejecting the parameters if it gives a flux that is too low : {reject_low_flux}")
    print(f"Confidence used for calculating the error : confidence = {confidence_option} sigma")
    print("======================================================================================")
    print(f"Parameters used for the simulations and the number of values if a distribution is used")
    n_distri = 60
    distri_nu = "distri_pearce"
    gamma_option = 100
    print(f"Values for gamma = {gamma_option}")
    # red_z_option = ("distri", n_distri)
    red_z_option = 1
    print(f"Values for redshift = {red_z_option}")
    theta_j_option = ("distri", n_distri)
    theta_nu_option = (distri_nu, n_distri)
    print(f"Values for theta j = {theta_j_option}")
    print(f"Values for theta nu = {theta_nu_option}")
    # nu_0_option = ("distri", n_distri)
    nu_0_option = 350 / 100
    print(f"Values for nu 0 = {nu_0_option}")
    alpha_option = -0.8
    beta_option = -2.2
    print(f"Values for alpha = {alpha_option}")
    print(f"Values for beta = {beta_option}")
    print("======================================================================================")
    print("initialisation")

    test_distri_SO = PolVSAngleRatio(model="SO", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, jet_model=jet_model_used, flux_rejection=reject_low_flux, integ_steps=int_step,
                                  confidence=confidence_option, parallel=True, special_mode=None)
    print("calcul")
    test_distri_SO.pf_calculation()

    test_distri_SR = PolVSAngleRatio(model="SR", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, jet_model=jet_model_used, flux_rejection=reject_low_flux, integ_steps=int_step,
                                  confidence=confidence_option, parallel=True, special_mode=None)
    print("calcul")
    test_distri_SR.pf_calculation()

    test_distri_CD = PolVSAngleRatio(model="CD", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, jet_model=jet_model_used, flux_rejection=reject_low_flux, integ_steps=int_step,
                                  confidence=confidence_option, parallel=True, special_mode=None)
    print("calcul")
    test_distri_CD.pf_calculation()

    test_distri_PJ = PolVSAngleRatio(model="PJ", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, jet_model=jet_model_used, flux_rejection=reject_low_flux, integ_steps=int_step,
                                  confidence=confidence_option, parallel=True, special_mode=None)
    print("calcul")
    test_distri_PJ.pf_calculation()
    print("display")

    # Values of histograms given by pearce
    x_SO = np.array([0.006646525679758308, 0.113595166163142, 0.12507552870090635, 0.13716012084592144, 0.1486404833836858, 0.1607250755287009,
            0.172809667673716, 0.18429003021148035, 0.19637462235649547, 0.20785498489425983, 0.22054380664652568, 0.23202416918429003,
            0.24410876132930515, 0.25619335347432026, 0.2676737160120846, 0.2797583081570997, 0.29123867069486403, 0.30332326283987915,
            0.3148036253776435, 0.3268882175226586, 0.338368580060423, 0.3504531722054381, 0.36253776435045315, 0.37462235649546827,
            0.3861027190332326, 0.39818731117824774, 0.4096676737160121, 0.4217522658610272, 0.43323262839879156, 0.4453172205438066,
            0.45740181268882174, 0.4688821752265861, 0.4809667673716012, 0.49244712990936557])
    y_SO = np.array([0.019230769230769232, 0, 0, 0, 0, 0, 0, 0, 0, 0.038461538461538464, 0.019230769230769232, 0.038461538461538464,
            0.038461538461538464, 0.019230769230769232, 0.31730769230769235, 0.1875, 0.4278846153846154, 0.16826923076923078, 0.125,
            0.038461538461538464, 0.08653846153846154, 0.125, 0.0625, 0.08653846153846154, 0.038461538461538464, 0.10576923076923078,
            0.1875, 0.08173076923076923, 0.10576923076923078, 0.2932692307692308, 0.42307692307692313, 1, 0.46634615384615385,
            0.25480769230769235, 0.23076923076923078, 0.16826923076923078, 0.23076923076923078, 0.0625, 0.08173076923076923,
            0.08173076923076923, 0.038461538461538464, 0.019230769230769232, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x_SR = np.array([0.0072507552870090634, 0.01812688821752266, 0.030211480362537763, 0.04229607250755287, 0.054380664652567974,
            0.06646525679758308, 0.07794561933534744, 0.09003021148036254, 0.10151057401812688, 0.113595166163142, 0.12507552870090635,
            0.13716012084592144, 0.14924471299093656, 0.1607250755287009, 0.17341389728096676, 0.18429003021148035, 0.19637462235649547,
            0.2084592145015106, 0.22054380664652568, 0.23202416918429003, 0.24410876132930515, 0.2555891238670695, 0.2676737160120846,
            0.2791540785498489, 0.29123867069486403, 0.3027190332326284, 0.3148036253776435])
    y_SR = np.array([1., 0.1764705882352941, 0.1568627450980392, 0.09313725490196079, 0.029411764705882353, 0.0392156862745098, 0.058823529411764705,
            0.058823529411764705, 0.06372549019607843, 0.029411764705882353, 0.06862745098039216, 0.029411764705882353,
            0.029411764705882353, 0.049019607843137254, 0.029411764705882353, 0.1323529411764706, 0.09803921568627451, 0.049019607843137254,
            0.0392156862745098, 0.0392156862745098, 0.058823529411764705, 0.0392156862745098, 0.049019607843137254, 0.049019607843137254,
            0.0392156862745098, 0.0392156862745098, 0.014705882352941176, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    x_CD = np.array([0.007854984894259818, 0.01812688821752266, 0.030211480362537763, 0.04229607250755287, 0.054380664652567974, 0.06646525679758308,
            0.07794561933534744, 0.08942598187311178, 0.10151057401812688, 0.113595166163142, 0.12507552870090635, 0.13716012084592144,
            0.14924471299093656, 0.1607250755287009, 0.17220543806646527, 0.18489425981873112, 0.1957703927492447, 0.20785498489425983,
            0.22054380664652568, 0.23202416918429003, 0.24350453172205438, 0.25619335347432026, 0.2676737160120846, 0.2797583081570997,
            0.29063444108761327, 0.30332326283987915, 0.31540785498489426, 0.3268882175226586, 0.338368580060423, 0.3504531722054381,
            0.36253776435045315, 0.3740181268882175, 0.3867069486404834, 0.39818731117824774, 0.4096676737160121, 0.4217522658610272,
            0.43323262839879156, 0.4453172205438066, 0.456797583081571, 0.4688821752265861, 0.4809667673716012, 0.49244712990936557,
            0.5045317220543807, 0.5166163141993958, 0.5287009063444109, 0.5401812688821752])
    y_CD = np.array([1., 0.1553398058252427, 0.09223300970873785, 0.1553398058252427, 0.06310679611650485, 0.08252427184466019, 0.029126213592233007,
            0.019417475728155338, 0.009708737864077669, 0.02427184466019417, 0.02427184466019417, 0.029126213592233007,
            0.029126213592233007, 0.038834951456310676, 0.029126213592233007, 0.04854368932038834, 0.019417475728155338,
            0.009708737864077669, 0.009708737864077669, 0.038834951456310676, 0.029126213592233007, 0.02427184466019417,
            0.029126213592233007, 0.009708737864077669, 0.009708737864077669, 0.029126213592233007, 0.029126213592233007,
            0.019417475728155338, 0.07766990291262135, 0.04854368932038834, 0.06796116504854369, 0.058252427184466014, 0.029126213592233007,
            0.019417475728155338, 0.038834951456310676, 0.019417475728155338, 0.04854368932038834, 0.02427184466019417,
            0.029126213592233007, 0.019417475728155338, 0.029126213592233007, 0.019417475728155338, 0.04854368932038834,
            0.029126213592233007, 0.019417475728155338, 0.009708737864077669, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    x_PJ = np.array([0.0072507552870090634, 0.018731117824773415, 0.03081570996978852, 0.04229607250755287, 0.05377643504531722, 0.06646525679758308,
            0.07794561933534744, 0.09003021148036254, 0.10211480362537764, 0.113595166163142, 0.12567975830815709, 0.13716012084592144,
            0.14924471299093656, 0.1607250755287009, 0.172809667673716, 0.18489425981873112, 0.19637462235649547, 0.2084592145015106,
            0.2199395770392749, 0.2326283987915408, 0.24410876132930515, 0.25619335347432026, 0.2676737160120846, 0.2797583081570997,
            0.29123867069486403, 0.3039274924471299, 0.31540785498489426, 0.3268882175226586, 0.338368580060423, 0.34984894259818733,
            0.3631419939577039, 0.37462235649546827, 0.3867069486404834, 0.39818731117824774])
    y_PJ = np.array([1., 0.1, 0.135, 0.155, 0.135, 0.065, 0.1, 0.065, 0.085, 0.105, 0.17, 0.15, 0.045, 0.085, 0.045, 0.1, 0.08, 0.12, 0.085,
            0.015, 0.12, 0.1, 0.03, 0.05, 0.1, 0.03, 0.03, 0.105, 0.155, 0.115, 0.185, 0.135, 0.155, 0.44, 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    calc_bins = np.linspace(0, 0.7, 60)
    x_pearce = (calc_bins[1:] + calc_bins[:-1]) / 2
    fig_comp, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 10), sharex="all")

    hist1 = ax1.hist(test_distri_SO.data.pf, bins=calc_bins, label="Distribution of PF for SO model",
                     weights=[1 / test_distri_SO.data.n_val] * test_distri_SO.data.n_val, color='blue')
    ax1.scatter(x_pearce, y_SO / np.sum(y_SO), label="Values from Pearce")
    ax1.set(title='Distribution of Polarization fraction')
    ax1.legend()

    hist2 = ax2.hist(test_distri_SR.data.pf, bins=calc_bins, label="Distribution of PF for SR model",
                     weights=[1 / test_distri_SR.data.n_val] * test_distri_SR.data.n_val, color='red')
    ax2.scatter(x_pearce, y_SR / np.sum(y_SR), label="Values from Pearce")
    ax2.legend()

    hist3 = ax3.hist(test_distri_CD.data.pf, bins=calc_bins, label="Distribution of PF for CD model",
                     weights=[1 / test_distri_CD.data.n_val] * test_distri_CD.data.n_val, color='green')
    ax3.scatter(x_pearce, y_CD / np.sum(y_CD), label="Values from Pearce")
    ax3.set(ylabel='Proportion of occurence')
    ax3.legend()

    hist4 = ax4.hist(test_distri_PJ.data.pf, bins=calc_bins, label="Distribution of PF for PJ model",
                     weights=[1 / test_distri_PJ.data.n_val] * test_distri_PJ.data.n_val, color='orange')
    ax4.scatter(x_pearce, y_PJ / np.sum(y_PJ), label="Values from Pearce")
    ax4.set(xlabel='Polarization fraction', xlim=(0, 0.75), xticks=np.arange(0, 0.7, 0.15))
    ax4.legend()

########################################################################################################################
# test adequation of PJ distri with values from Pearce (mainly for searching the right limit flux)                     #
########################################################################################################################
activate_PJ_distrib = False
if activate_PJ_distrib:
    print("======================================================================================")
    print("Paramètres de simulation de la distribution")
    print("======================================================================================")
    int_step = 100
    jet_model_used = "top-hat"
    reject_low_flux = True
    confidence_option = 1.96
    print(f"Number of iterations for calculating the integrals : int_step = {int_step}")
    print(f"Model used for the jet : {jet_model_used}")
    print(f"Rejecting the parameters if it gives a flux that is too low : {reject_low_flux}")
    print(f"Confidence used for calculating the error : confidence = {confidence_option} sigma")
    print("======================================================================================")
    print(f"Parameters used for the simulations and the number of values if a distribution is used")
    n_distri = 100
    distri_nu = "distri_pearce"
    gamma_option = 100
    print(f"Values for gamma = {gamma_option}")
    # red_z_option = ("distri", n_distri)
    red_z_option = 1
    print(f"Values for redshift = {red_z_option}")
    theta_j_option = ("distri", n_distri)
    theta_nu_option = (distri_nu, n_distri)
    print(f"Values for theta j = {theta_j_option}")
    print(f"Values for theta nu = {theta_nu_option}")
    # nu_0_option = ("distri", n_distri)
    nu_0_option = 350 / 100
    print(f"Values for nu 0 = {nu_0_option}")
    alpha_option = -0.8
    beta_option = -2.2
    print(f"Values for alpha = {alpha_option}")
    print(f"Values for beta = {beta_option}")
    print("======================================================================================")
    print("initialisation")

    test_distri_PJ = PolVSAngleRatio(model="PJ", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, jet_model=jet_model_used, flux_rejection=reject_low_flux, integ_steps=int_step,
                                  confidence=confidence_option, parallel=True, special_mode=None)
    print("calcul")
    test_distri_PJ.pf_calculation()
    print("display")

    y_PJ = np.array([1., 0.1, 0.135, 0.155, 0.135, 0.065, 0.1, 0.065, 0.085, 0.105, 0.17, 0.15, 0.045, 0.085, 0.045, 0.1, 0.08, 0.12, 0.085,
            0.015, 0.12, 0.1, 0.03, 0.05, 0.1, 0.03, 0.03, 0.105, 0.155, 0.115, 0.185, 0.135, 0.155, 0.44, 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    calc_bins = np.linspace(0, 0.7, 60)
    x_pearce = (calc_bins[1:] + calc_bins[:-1]) / 2

    fig_comp, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    hist1 = ax1.hist(test_distri_PJ.data.pf, bins=calc_bins, label="Distribution of PF for PJ model",
                     weights=[1 / test_distri_PJ.data.n_val] * test_distri_PJ.data.n_val, color='orange')
    ax1.scatter(x_pearce, y_PJ / np.sum(y_PJ), label="Values from Pearce")
    ax1.set(title='Distribution of Polarization fractionPJ model', xlabel='Polarization fraction', xlim=(0, 0.7),
            xticks=np.arange(0, 0.75, 0.15))
    ax1.legend()

    fig_comp.show()

########################################################################################################################
# test toma curves                                                                                                     #
########################################################################################################################
Acti_toma = False
if Acti_toma:
    ######################################################################################################################
    # test toma curves                                                                                                   #
    ######################################################################################################################
    print("======================================================================================")
    print("Paramètres de simulation de la distribution")
    print("======================================================================================")
    int_step = 100
    print(f"Number of iterations for calculating the integrals : int_step = {int_step}")
    confidence_option = 1.96
    print(f"Confidence used for calculating the error : confidence = {confidence_option} sigma")
    print("======================================================================================")
    print(f"Parameters used for the simulations and the number of values if a distribution is used")
    n_distri = 10
    distri_nu = "distri_pearce"
    gamma_option = 100
    print(f"Values for gamma = {gamma_option}")
    # red_z_option = ("distri", n_distri)
    red_z_option = 1
    print(f"Values for redshift = {red_z_option}")
    theta_j_option = list(np.sqrt([0.1, 1, 10, 100]) / 100)
    theta_nu_option = (0.001, 5, 100)
    print(f"Values for theta j = {theta_j_option}")
    print(f"Values for theta nu = {theta_nu_option}")
    # nu_0_option = ("distri", n_distri)
    nu_0_option = 350 / 100
    print(f"Values for nu 0 = {nu_0_option}")
    alpha_option = -0.8
    beta_option = -2.2
    print(f"Values for alpha = {alpha_option}")
    print(f"Values for beta = {beta_option}")
    print("======================================================================================")
    print("initialisation")
    test_distri_SO = PolVSAngleRatio(model="SO", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, integ_steps=int_step, confidence=confidence_option, parallel=True,
                                  special_mode="Toma")
    print("calcul")
    valtime = time()
    test_distri_SO.pf_calculation()
    print("Time taken : ", time() - valtime)
    print("display")
    test_distri_SO.toma_display()

    test_distri_SR = PolVSAngleRatio(model="SR", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, integ_steps=int_step, confidence=confidence_option, parallel=True,
                                  special_mode="Toma")
    print("calcul")
    valtime = time()
    test_distri_SR.pf_calculation()
    print("Time taken : ", time() - valtime)
    print("display")
    test_distri_SR.toma_display()

    test_distri_CD = PolVSAngleRatio(model="CD", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, integ_steps=int_step, confidence=confidence_option, parallel=True,
                                  special_mode="Toma")
    print("calcul")
    valtime = time()
    test_distri_CD.pf_calculation()
    print("Time taken : ", time() - valtime)
    print("display")
    test_distri_CD.toma_display()

    test_distri_PJ = PolVSAngleRatio(model="PJ", gamma_range=gamma_option, red_z_range=red_z_option, theta_j_range=theta_j_option,
                                  theta_nu_range=theta_nu_option, nu_0_range=nu_0_option, alpha_range=alpha_option, beta_range=beta_option,
                                  nu_min=None, nu_max=None, integ_steps=int_step, confidence=confidence_option, parallel=True,
                                  special_mode="Toma")
    print("calcul")
    valtime = time()
    test_distri_PJ.pf_calculation()
    print("Time taken : ", time() - valtime)
    print("display")
    test_distri_PJ.toma_display()

########################################################################################################################
# tests upgraded integral calculation                                                                                                     #
########################################################################################################################
Acti_test = False
if Acti_test:
    diff = 0
    steps = 100
    counts = 1000
    list_vals = []
    for i in range(counts):
        # init_time = time()
        # v1 = models.integral_calculation_SO(60, 500, 100, 1, 0.0316227766016838, 0.0316227766016838, 3.5, -0.2, 1.2, integ_steps=steps)
        # print(v1)
        # print("calc time for SO unoptimized version : ", time()-init_time)
        init_time = time()
        # v2 = models.integral_calculation_SO_opti(60, 500, 100, 1, 0.0316227766016838, 0.0316227766016838, 3.5, -0.2, 1.2, integ_steps=steps)
        v2 = integral_calculation_SO(60, 500, 100, 1, 0.1, 0.1, 3.5, -0.2, 1.2, integ_steps=steps)
        print(v2)
        print("calc time for SO optimized version : ", time() - init_time)
        list_vals.append(v2[0])
        # print("========================================================")
        # init_time = time()
        # print(models.integral_calculation_SR(60, 500, 100, 1, 0.0316227766016838, 0.0316227766016838, 3.5, -0.2, 1.2, integ_steps=steps))
        # print("calc time for SR unoptimized version : ", time()-init_time)
        # init_time = time()
        # print(models.integral_calculation_SR_opti(60, 500, 100, 1, 0.0316227766016838, 0.0316227766016838, 3.5, -0.2, 1.2, integ_steps=steps))
        # print("calc time for SR optimized version : ", time()-init_time)
        # print("========================================================")
        # init_time = time()
        # print(models.integral_calculation_CD(60, 500, 100, 1, 0.0316227766016838, 0.0316227766016838, 3.5, -0.2, 1.2, integ_steps=steps))
        # print("calc time for CD unoptimized version : ", time()-init_time)
        # init_time = time()
        # print(models.integral_calculation_CD_opti(60, 500, 100, 1, 0.0316227766016838, 0.0316227766016838, 3.5, -0.2, 1.2, integ_steps=steps))
        # print("calc time for CD optimized version : ", time()-init_time)
    std = np.sqrt(np.sum((list_vals - np.mean(list_vals)) ** 2) / len(list_vals))
    print(std)
    print(test_distri_SO.data.pf[640])
    print(test_distri_SO.data.error_pf[640])
    a = [1, 1]
    b = [1, 1, 1, 1]
    c = [1, 1, 1]

    d = np.reshape(a, (len(a), 1, 1)) * np.reshape(b, (len(b), 1)) * c

#############
# Pourrait être intéressant de tester à partir de quand est ce que l'erreur est assez faible
# Donner une indication du temps que ça prendrait aussi pour 1 ite avec telle précision et tant de coeurs
# Voir comment est-ce qu'on pourrait intégrer une erreur à un histogramme
# Mettre d'autres modèles et s'assurer que tout est correct !
# S'assurer du fonctionnement de tout ça pour un z != 1 !!!
#       >> voir ce que le z change vraiment et voir si la correction apporte quelque chose

# Flux limite détectable
# Flux modifié de la source par rapport à l'angle de réception (voir si on peut pas chopper une distribution ???)
# Si distribution est-ce qu'il faut prendre en compte le fait que la distribution intègre peut être déjà l'angle d'ouverture ?...
# Faire une dépendance à PF pour le flux ? (plus simple de détecter une PF élevée)
#       Dans ce cas on aurait ce que devrait vraiment détecter le sursaut, ça reviendrait à mettre une coupure ?...
# Faraday depolarisation négligée dans Toma mais il est dit que ça peut avoir un effet important !
# Une fois que le cut sur la luminosité est appliqué, faire varier le facteur de lorentz aussi ?
#   pas sûr de ça, les modèles sont faits pour des top-hat peut être ? donc je fait d'avoir un gamma qui varie
#   peut être intéressant mais peut êrte conflict avec le modèle ?

# Calculer les flux de GBM (en utilisant la fluence et le t90 j'imagine), doute sur le fait qu'il faille prendre en compte le fait de
# peut etre voir le susaut de coté...
# Ou alors simplement trouver une valeur de luminosité généralement admise pour les sursauts (10^52 erg/s ?)
# Trouver le flux limite de détection, celui de pearce puis en avoir un pour COMCUBE
