# Regular imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from itertools import repeat

# Developped modules imports
from src.General.funcmod import arg_convert, var_ite_setting, values_number
from src.Polarization.models import integral_calculation_SO, integral_calculation_SR, integral_calculation_CD, integral_calculation_PJ

import matplotlib as mpl
mpl.use('Qt5Agg')


class PolVSAngleRatio:
    
    def __init__(self, model=None, gamma_range=None, red_z_range=None, theta_j_range=None, theta_nu_range=None, nu_0_range=None,
                 alpha_range=None, beta_range=None, nu_min=None, nu_max=None, jet_model=None, flux_rejection=False, integ_steps=None,
                 confidence=None, parallel="all"):
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
        Method that creates the parameter used to obtain a set of polarization
        """
        print("Creation of parameters list")
        init_time = time.time()
        arg_list = []
        ite_count = 0

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
                                    arg_list.append(var_ite_setting(ite_count, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta,
                                                                    self.jet_model, self.flux_rejection))
                                    ite_count += 1
        arg_list = np.array(arg_list)
        print(f"Creation finished after {time.time()-init_time} seconds")
        return arg_list

    def integral_calculation(self, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta):
        """
        Method In which one need to insert the different model he wants to use.
        Models are defined in the "models.py" module
        """
        if self.model == "SO":
            # print(self.nu_min, self.nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, self.integ_steps, self.confidence)
            return integral_calculation_SO(self.nu_min, self.nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta,
                                           integ_steps=self.integ_steps, confidence=self.confidence)[:2]
        elif self.model == "SR":
            return integral_calculation_SR(self.nu_min, self.nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta,
                                           integ_steps=self.integ_steps, confidence=self.confidence)
        elif self.model == "CD":
            return integral_calculation_CD(self.nu_min, self.nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta,
                                           integ_steps=self.integ_steps, confidence=self.confidence)
        elif self.model == "PJ":
            return integral_calculation_PJ(gamma, theta_j, theta_nu)

    def pf_calculation(self, param_list, timer_info=None):
        """
        Method used to calculate the polarization fraction using a multi processing method
        Returns the arguments to be saved in the form of a list :
        [iteration, pf, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta]
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
        Calculation method that first set the data container, create the parameters and then calculate and stores the pf and parameters used
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

                # ax1.plot(x_list, y_list - errors, color=colors_err[yj_index])
                # ax1.plot(x_list, y_list + errors, color=colors_err[yj_index])
                ax1.set(xlabel=r'q=$\theta_{\nu}$/$\theta_j$', ylabel='PF',
                        title=f"Model {self.model}\n" + r'Variation of PF as fonction of q=$\theta_{\nu}$/$\theta_j$ using different values of yj',
                        ylim=(0, 1))
            ax1.legend()
            figure.show()

# simtime = time.time()
# for model in ["SO", "SR", "CD", "PJ"]:
# # for model in ["SO", "SR"]:
#     test_distri = PolVSAngleRatio(model=model, gamma_range=100, red_z_range=1, theta_j_range=list(np.sqrt([0.1, 1, 10, 100]) / 100),
#                                      theta_nu_range="toma_curve", nu_0_range=3.5, alpha_range=-0.8, beta_range=-2.2,
#                                      nu_min=None, nu_max=None, integ_steps=200, confidence=1.96, parallel=40)
#     test_distri.toma_display()
# print(f"TIME TAKEN FOR 4 MODELS : {time.time() - simtime} s")


# print("======================================================================================")
# distri_nu = "distri_pearce"
#
# xlist_pearce = [np.array([0.006646525679758308, 0.113595166163142, 0.12507552870090635, 0.13716012084592144, 0.1486404833836858,
#                           0.1607250755287009, 0.172809667673716, 0.18429003021148035, 0.19637462235649547, 0.20785498489425983,
#                           0.22054380664652568, 0.23202416918429003, 0.24410876132930515, 0.25619335347432026, 0.2676737160120846,
#                           0.2797583081570997, 0.29123867069486403, 0.30332326283987915, 0.3148036253776435, 0.3268882175226586,
#                           0.338368580060423, 0.3504531722054381, 0.36253776435045315, 0.37462235649546827, 0.3861027190332326,
#                           0.39818731117824774, 0.4096676737160121, 0.4217522658610272, 0.43323262839879156, 0.4453172205438066,
#                           0.45740181268882174, 0.4688821752265861, 0.4809667673716012, 0.49244712990936557]),
#                 np.array([0.0072507552870090634, 0.01812688821752266, 0.030211480362537763, 0.04229607250755287, 0.054380664652567974,
#                           0.06646525679758308, 0.07794561933534744, 0.09003021148036254, 0.10151057401812688, 0.113595166163142,
#                           0.12507552870090635, 0.13716012084592144, 0.14924471299093656, 0.1607250755287009, 0.17341389728096676,
#                           0.18429003021148035, 0.19637462235649547, 0.2084592145015106, 0.22054380664652568, 0.23202416918429003,
#                           0.24410876132930515, 0.2555891238670695, 0.2676737160120846, 0.2791540785498489, 0.29123867069486403,
#                           0.3027190332326284, 0.3148036253776435]),
#                 np.array([0.007854984894259818, 0.01812688821752266, 0.030211480362537763, 0.04229607250755287, 0.054380664652567974,
#                           0.06646525679758308, 0.07794561933534744, 0.08942598187311178, 0.10151057401812688, 0.113595166163142,
#                           0.12507552870090635, 0.13716012084592144, 0.14924471299093656, 0.1607250755287009, 0.17220543806646527,
#                           0.18489425981873112, 0.1957703927492447, 0.20785498489425983, 0.22054380664652568, 0.23202416918429003,
#                           0.24350453172205438, 0.25619335347432026, 0.2676737160120846, 0.2797583081570997, 0.29063444108761327,
#                           0.30332326283987915, 0.31540785498489426, 0.3268882175226586, 0.338368580060423, 0.3504531722054381,
#                           0.36253776435045315, 0.3740181268882175, 0.3867069486404834, 0.39818731117824774, 0.4096676737160121,
#                           0.4217522658610272, 0.43323262839879156, 0.4453172205438066, 0.456797583081571, 0.4688821752265861,
#                           0.4809667673716012, 0.49244712990936557, 0.5045317220543807, 0.5166163141993958, 0.5287009063444109,
#                           0.5401812688821752]),
#                 np.array([0.0072507552870090634, 0.018731117824773415, 0.03081570996978852, 0.04229607250755287, 0.05377643504531722,
#                           0.06646525679758308, 0.07794561933534744, 0.09003021148036254, 0.10211480362537764, 0.113595166163142,
#                           0.12567975830815709, 0.13716012084592144, 0.14924471299093656, 0.1607250755287009, 0.172809667673716,
#                           0.18489425981873112, 0.19637462235649547, 0.2084592145015106, 0.2199395770392749, 0.2326283987915408,
#                           0.24410876132930515, 0.25619335347432026, 0.2676737160120846, 0.2797583081570997, 0.29123867069486403,
#                           0.3039274924471299, 0.31540785498489426, 0.3268882175226586, 0.338368580060423, 0.34984894259818733,
#                           0.3631419939577039, 0.37462235649546827, 0.3867069486404834, 0.39818731117824774])]
#
# ylist_pearce = [np.array([0.019230769230769232, 0, 0, 0, 0, 0, 0, 0, 0, 0.038461538461538464, 0.019230769230769232, 0.038461538461538464,
#                           0.038461538461538464, 0.019230769230769232, 0.31730769230769235, 0.1875, 0.4278846153846154, 0.16826923076923078,
#                           0.125, 0.038461538461538464, 0.08653846153846154, 0.125, 0.0625, 0.08653846153846154, 0.038461538461538464,
#                           0.10576923076923078, 0.1875, 0.08173076923076923, 0.10576923076923078, 0.2932692307692308, 0.42307692307692313,
#                           1, 0.46634615384615385, 0.25480769230769235, 0.23076923076923078, 0.16826923076923078, 0.23076923076923078,
#                           0.0625, 0.08173076923076923, 0.08173076923076923, 0.038461538461538464, 0.019230769230769232, 0, 0, 0, 0, 0, 0,
#                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#                 np.array([1., 0.1764705882352941, 0.1568627450980392, 0.09313725490196079, 0.029411764705882353, 0.0392156862745098,
#                           0.058823529411764705, 0.058823529411764705, 0.06372549019607843, 0.029411764705882353, 0.06862745098039216,
#                           0.029411764705882353, 0.029411764705882353, 0.049019607843137254, 0.029411764705882353, 0.1323529411764706,
#                           0.09803921568627451, 0.049019607843137254, 0.0392156862745098, 0.0392156862745098, 0.058823529411764705,
#                           0.0392156862745098, 0.049019607843137254, 0.049019607843137254,  0.0392156862745098, 0.0392156862745098,
#                           0.014705882352941176, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#                           0., 0., 0., 0., 0., 0., 0., 0., 0.]),
#                 np.array([1., 0.1553398058252427, 0.09223300970873785, 0.1553398058252427, 0.06310679611650485, 0.08252427184466019,
#                           0.029126213592233007, 0.019417475728155338, 0.009708737864077669, 0.02427184466019417, 0.02427184466019417,
#                           0.029126213592233007, 0.029126213592233007, 0.038834951456310676, 0.029126213592233007, 0.04854368932038834,
#                           0.019417475728155338, 0.009708737864077669, 0.009708737864077669, 0.038834951456310676, 0.029126213592233007,
#                           0.02427184466019417, 0.029126213592233007, 0.009708737864077669, 0.009708737864077669, 0.029126213592233007,
#                           0.029126213592233007, 0.019417475728155338, 0.07766990291262135, 0.04854368932038834, 0.06796116504854369,
#                           0.058252427184466014, 0.029126213592233007, 0.019417475728155338, 0.038834951456310676, 0.019417475728155338,
#                           0.04854368932038834, 0.02427184466019417, 0.029126213592233007, 0.019417475728155338, 0.029126213592233007,
#                           0.019417475728155338, 0.04854368932038834, 0.029126213592233007, 0.019417475728155338, 0.009708737864077669,
#                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
#                 np.array([1., 0.1, 0.135, 0.155, 0.135, 0.065, 0.1, 0.065, 0.085, 0.105, 0.17, 0.15, 0.045, 0.085, 0.045, 0.1, 0.08, 0.12,
#                           0.085, 0.015, 0.12, 0.1, 0.03, 0.05, 0.1, 0.03, 0.03, 0.105, 0.155, 0.115, 0.185, 0.135, 0.155, 0.44, 0., 0., 0.,
#                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]
#
# list_distri = []
# n_distri = 100
# int_steps = 100
# for distname in ["SO", "SR", "CD", "PJ"]:
# # for distname in ["SO"]:
#     simtime = time.time()
#     list_distri.append(PolVSAngleRatio(model=distname, gamma_range=100, red_z_range=1, theta_j_range=("distri", n_distri),
#                                   theta_nu_range=(distri_nu, n_distri), nu_0_range=350 / 100, alpha_range=-0.8, beta_range=-2.2,
#                                   nu_min=None, nu_max=None, jet_model="top-hat", flux_rejection=True, integ_steps=int_steps,
#                                   confidence=1.96, parallel=10))
#     print(f"TIME TAKEN FOR {distname} : {time.time() - simtime} s")
#
#
# bins = np.linspace(0, 0.7, 60)
# x_pearce = (bins[1:] + bins[:-1]) / 2
#
# labels = ["Distribution of PF for SO model", "Distribution of PF for SR model",
#           "Distribution of PF for CD model", "Distribution of PF for PJ model"]
#
# colors = ['blue', 'red', 'green', 'orange']
# fig_comp, axes = plt.subplots(len(list_distri), 1, figsize=(20, 10), sharex="all")
# fig_comp.suptitle(f"Distribution of Polarization fraction\nInt step : {int_steps}, number of pf estimated : {n_distri**2}")
# if len(list_distri) == 1:
#     axes = [axes]
# for ax_idx in range(len(axes)):
#     axes[ax_idx].hist(list_distri[ax_idx].data_df.pf.values, bins=bins, label=labels[ax_idx],
#                       weights=[1 / len(list_distri[ax_idx].data_df)] * len(list_distri[ax_idx].data_df), color=colors[ax_idx])
#     axes[ax_idx].scatter(x_pearce, ylist_pearce[ax_idx] / np.sum(ylist_pearce[ax_idx]), label="Values from Pearce")
#     axes[ax_idx].legend()
#
# axes[-1].set(xlabel='Polarization fraction', xlim=(0, 0.75), xticks=np.arange(0, 0.7, 0.15))
# plt.show()



# # VERIFICATION INTEGRAL CALCULATION
# # integral_calculation_SO2(nu_min, nu_max, gamma, red_z, theta_j, theta_nu, nu_0, alpha, beta, integ_steps=70, confidence=1.96)
# intsteps = 70
# seed = True
# init_time = time.time()
# oldvals = integral_calculation_SO_old(60, 500, 100.0, 1.0, 0.011130993529999907, 0.01683407092870492, 3.5, -0.2, 1.2, intsteps, 1.96, seed)
# print(f"VAL OPTI : {oldvals[0]} +- {oldvals[1]}")
# print("time ref : ", time.time() - init_time)
# init_time = time.time()
# testref = integral_calculation_SO3(60, 500, 100.0, 1.0, 0.011130993529999907, 0.01683407092870492, 3.5, -0.2, 1.2, intsteps, 1.96, seed)
# print("\nVAL REF : ", testref[0])
# print("time no opti : ", time.time() - init_time)
# init_time = time.time()
# test = integral_calculation_SO(60, 500, 100.0, 1.0, 0.011130993529999907, 0.01683407092870492, 3.5, -0.2, 1.2, intsteps, 1.96, seed)
# print(f"\nVAL WORKING : {test[0]} +- {test[1]}")
# print("time small opti : ", time.time() - init_time)
# # print("\nREF VALUE = ", 0.48140914540800367)
#
# # tt = np.array([[[0.0031424782151871543, 0.004408650166616973, 0.00567526819880468, 0.0096809597687394], [0.1575910441482942, 0.3151820882965884, 0.4727731324448826, 0.6303641765931768], [0.002184678231189442, 0.0038764184221485518, 0.005566303951313738, 0.007819156801451808], [0.00313991115092023, 0.0043943703157195355, 0.005649343917696417, 0.009669264862208617]], [[0.0018600235364810305, 0.002609466959671577, 0.0033591744167398552, 0.005730131378017785], [0.10872299005896736, 0.21744598011793473, 0.3261689701769021, 0.43489196023586946], [0.0012931045663296982, 0.0022944405684656604, 0.0032946788017858388, 0.0046281357228979], [0.001858504098754124, 0.002601014758272596, 0.0033438299116309005, 0.005723209197523718]], [[0.009633726764270623, 0.013515362142904979, 0.017398365047371505, 0.02967839865275803], [0.22372190121465596, 0.4474438024293119, 0.6711657036439679, 0.8948876048586238], [0.007435834982662469, 0.013193891576040506, 0.01894563558807298, 0.026613511705796292], [0.00961846346545465, 0.01346123769235933, 0.01730558779035079, 0.02961977786782309]], [[0.004152494647129859, 0.005825623907103155, 0.007499342621587343, 0.012792493970041953], [0.17892525386036248, 0.35785050772072496, 0.5367757615810874, 0.7157010154414499], [0.002886850453528124, 0.005122328826406446, 0.007355356435064896, 0.01033229334929978], [0.004149102508856403, 0.005806754403369816, 0.007465086079087707, 0.012777040231482035]]]).flatten()
# # tt = np.array(testref[2]).flatten()
# # to = np.array(test[2]).flatten()
# # # print(test[2])
# #
# # for itei in range(len(tt)):
# #     print(f"comp val ref - calc: {tt[itei] == to[itei]}")
# # print(f"lens : {len(tt) == len(to)}")
# print(f"pf : {round(testref[0], 8) == round(test[0], 8)}")# : {testref[0]} == {test[0]}")
# # More precise ref : 0.4186017020492592