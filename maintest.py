import numpy as np

from dataclass import *
from time import time


init_time = time()
bkg = "./backgrounds/bkg"  # _background_sat0_0000_90.0_0.0.inc1.id1.extracted.tra"
param = "./test/polGBM.par"
erg = (100, 460)
arm = 80
test = AllSourceData(bkg, param, erg, arm, parallel=True)
test.make_const()
test.analyze()
print("=======================================")
print("processing time : ", time()-init_time, "seconds")
print("=======================================")
sim = test.alldata[0][0]
sim0 = test.alldata[0][0][0]
sim1 = test.alldata[0][0][1]
sim2 = test.alldata[0][0][2]
simcons = sim.const_data
print(sim0.mdp)
print(sim0.dec_sat_frame)
print(sim0.ra_sat_frame)
print(len(sim0.pol))
print(len(sim0.unpol))
sim0.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
colors = ["blue", "green", "red"]
bins = sim[0].bins
var_x = .5 * (bins[1:] + bins[:-1])
binw = bins[1:] - bins[:-1]
ylabel = "Number of counts (per degree)"
for i in range(3):
  histtemp = np.histogram(sim[i].pol, bins)[0] / binw
  histtempnp = np.histogram(sim[i].unpol, bins)[0] / binw
  histcorrtemp = histtemp / histtempnp * np.mean(histtempnp)
  ax1.step(var_x, histcorrtemp, color=colors[i], where="mid")
  ax1.set(xlabel="Azimuthal scatter angle (degree)", ylabel=ylabel, xlim=(-180, 180))

polsum = np.concatenate((sim0.pol, sim1.pol, sim2.pol))
unpolsum = np.concatenate((sim0.unpol, sim1.unpol, sim2.unpol))
histtot = np.histogram(polsum, bins)[0] / binw
histtotnp = np.histogram(unpolsum, bins)[0] / binw
histtotcorr = histtot / histtotnp * np.mean(histtotnp)
ax2.step(var_x, histtotcorr, color='black', where="mid")
ax2.set(xlabel="Azimuthal scatter angle (degree)", ylabel=ylabel, xlim=(-180, 180))

histsum = np.histogram(sim0.pol, bins)[0] / binw + np.histogram(sim1.pol, bins)[0] / binw + np.histogram(sim2.pol, bins)[0] / binw
histsumnp = np.histogram(sim0.unpol, bins)[0] / binw + np.histogram(sim1.unpol, bins)[0] / binw + np.histogram(sim2.unpol, bins)[0] / binw
histsumcorr = histsum / histsumnp * np.mean(histsumnp)
ax3.step(var_x, histsumcorr, color='black', where="mid")
ax3.set(xlabel="Azimuthal scatter angle (degree)", ylabel=ylabel, xlim=(-180, 180))

histcons = np.histogram(sim.const_data.pol, bins)[0] / binw
histconsnp = np.histogram(sim.const_data.unpol, bins)[0] / binw
histconscorr = histcons / histconsnp * np.mean(histconsnp)
ax4.step(var_x, histconscorr, color='black', where="mid")
ax4.set(xlabel="Azimuthal scatter angle (degree)", ylabel=ylabel, xlim=(-180, 180))

plt.show()



# set_bin = np.linspace(-180, 180, 21)
# set_trafile = test.alldata[0][3][0]
# test_values = set_trafile.arm
# set_trafile.pol.show(set_trafile.unpol)
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
# binw = set_bin[1:]-set_bin[:-1]
# binpos = 0.5 * (set_bin[1:] + set_bin[:-1])
# ax1.hist(set_trafile.pol, bins=set_bin)
# ax2.hist(set_trafile.unpol, bins=set_bin)
# pol = np.histogram(set_trafile.pol, set_bin)[0] / binw
# unpol = np.histogram(set_trafile.unpol, set_bin)[0] / binw
# ax3.scatter(binpos, pol / unpol * np.mean(unpol))
# xvar = np.linspace(-180, 180, 1000)
# yvar = 0.57911 - 0.231988*np.cos((2*(xvar-17.0009))*np.pi/180)
# ax3.plot(xvar, yvar, color="orange")
# ax3.scatter(binpos, modulation_func(binpos, *set_trafile.pol.fits[-2].popt))
# ax3.errorbar(binpos, pol / unpol * np.mean(unpol), yerr=set_trafile.pol.polarigram_error, fmt = 'none')
# print(f"Modulation:           {set_trafile.mu100}+-{set_trafile.mu100_err}")
# print(f"Polarization angle:   {set_trafile.pa}+-{set_trafile.pa_err}")
# plt.show()
# print("cov : ", set_trafile.pol.fits[-2].pcov)
# print("PA, mu, S : ", set_trafile.pol.fits[-2].popt)
# print(set_trafile.pol.polarigram_error)

#from time import sleep
#from itertools import repeat
#def functest(intro, timeval, message1, message2):
#  sleep(timeval)
#  liste.append(timeval)
  #return [timeval, message]
#  return f"{intro} {timeval}{message1}{message2}"
#intro = "il faut attendre un peu, environ"
#message1 = "secondes. "
#message2 = "On peut fermer le programme maintenant"
#message = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
#timeval = np.random.random(10)
#functest(timeval[0], message[0])
#init_time = time()
#valtest = list(map(functest, intro, timeval, message1, message2))
#print(valtest)
#print(time()-init_time)
#init_time = time()
#with mp.Pool() as pool:
#  val = pool.starmap(functest, zip(repeat(intro), timeval, repeat(message1), repeat(message2)))

#print(val)
#print(time()-init_time)
#print(list(zip([1], [2], [3],[1, 1, 1, 1, 1, 1, 1])))
#pool.starmap(AllSimData, zip(repeat(self.sim_prefix), range(self.n_source), repeat(cat_data), repeat(self.mode), repeat(self.n_sim), repeat(self.sat_info), repeat(self.pol_data), repeat(self.sim_duration), repeat(self.options))


####################################################
# Test des fonctions d'angle
#   def angle(c, theta, phi):
#     """
#     Calculate the azimuthal Compton angle : Transforms the compton scattered gamma-ray vector (initialy in sat frame) into
#     a new referential corresponding to the direction of the source. In that frame c0 and c1 are the coordinates of
#     the vector in the plan orthogonal to the source direction. The x coordinate is the vector in the plane created by
#     the zenith (z axis) of the instrument and the source direction and y is in the plane of the detector (zcoord=0)
#     (so x is in the plane containing the zworld, source direction, and the axis yprime of the detector)
#     The way the azimuthal scattering angle is calculated imply that the polarization vector is colinear with x
#     Calculates the polar Compton angle
#     :param c:     3-uple, Compton scattered gamma-ray vector
#     :param theta: float,  source polar angle in sky in rad
#     :param phi:   float,  source azimuthal angle in sky in rad
#     :returns:     float,  angle in deg
#     """
#     theta, phi = np.deg2rad(theta), np.deg2rad(phi)
#     # Pluging in some MEGAlib magic
#     c = c / np.linalg.norm(c)
#     # c = (np.cos(-phi) * c[0] - np.sin(-phi) * c[1], np.sin(-phi) * c[0] + np.cos(-phi) * c[1], c[2])
#     # c = (np.sin(-theta) * c[2] + np.cos(-theta) * c[0], c[1], np.cos(-theta) * c[2] - np.sin(-theta) * c[0])
#     mat1 = np.array([[np.cos(-phi), - np.sin(-phi), 0],
#                      [np.sin(-phi), np.cos(-phi), 0],
#                      [0, 0, 1]])
#     mat2 = np.array([[np.cos(-theta), 0, np.sin(-theta)],
#                      [0, 1, 0],
#                      [- np.sin(-theta), 0, np.cos(-theta)]])
#     c = np.dot(mat1, c)
#     c = np.dot(mat2, c)
#     # Figure out a good arctan
#     polar = np.rad2deg(np.arccos(c[2]))
#     if c[0] > 0:
#       return np.arctan(c[1] / c[0]) * 180 / np.pi, polar
#     elif c[0] == 0:
#       return 90, polar
#     else:
#       if c[1] > 0:
#         return np.arctan(c[1] / c[0]) * 180 / np.pi + 180, polar
#       else:
#         return np.arctan(c[1] / c[0]) * 180 / np.pi - 180, polar
#
#
#   def angle2(c, theta, phi, source_name, num_sim, num_sat):
#     """
#     Calculate the azimuthal Compton angle : Transforms the compton scattered gamma-ray vector (initialy in sat frame) into
#     a new referential corresponding to the direction of the source. In that frame c0 and c1 are the coordinates of
#     the vector in the plan orthogonal to the source direction. The x coordinate is the vector in the plane created by
#     the zenith (z axis) of the instrument and the source direction and y is in the plane of the detector (zcoord=0)
#     (so x is in the plane containing the zworld, source direction, and the axis yprime of the detector)
#     The way the azimuthal scattering angle is calculated imply that the polarization vector is colinear with x
#     Calculates the polar Compton angle
#     :param c:     3-uple, Compton scattered gamma-ray vector
#     :param theta: float,  source polar angle in sky in deg
#     :param phi:   float,  source azimuthal angle in sky in deg
#     :returns:     float,  angle in deg
#     """
#     # print(theta, phi)
#     if len(c) == 0:
#       print(f"There is no compton event detected for source {source_name}, simulation {num_sim} and satellite {num_sat}")
#       return np.array([]), np.array([])
#     theta, phi = np.deg2rad(theta), np.deg2rad(phi)
#     # Pluging in some MEGAlib magic
#     c = c / np.reshape(np.linalg.norm(c, axis=1), (len(c), 1))
#     mat1 = np.array([[np.cos(-phi), - np.sin(-phi), 0],
#                      [np.sin(-phi), np.cos(-phi), 0],
#                      [0, 0, 1]])
#     mat2 = np.array([[np.cos(-theta), 0, np.sin(-theta)],
#                      [0, 1, 0],
#                      [- np.sin(-theta), 0, np.cos(-theta)]])
#     # using matrix products to combine the matrix instead of doing it vector by vector
#     c = np.matmul(c, np.transpose(mat1))
#     c = np.matmul(c, np.transpose(mat2))
#     # c = (np.cos(-phi) * c[0] - np.sin(-phi) * c[1], np.sin(-phi) * c[0] + np.cos(-phi) * c[1], c[2])
#     # c = (np.sin(-theta) * c[2] + np.cos(-theta) * c[0], c[1], np.cos(-theta) * c[2] - np.sin(-theta) * c[0])
#     polar = np.rad2deg(np.arccos(c[:, 2]))
#     # Figure out a good arctan
#     azim = np.where(c[:, 0] > 0, np.rad2deg(np.arctan(c[:, 1] / c[:, 0])), np.where(c[:, 0] == 0, 90, np.where(c[:, 1] > 0, np.rad2deg(np.arctan(c[:, 1] / c[:, 0])) + 180, np.rad2deg(np.arctan(c[:, 1] / c[:, 0])) - 180)))
#     return azim, polar
#
#
#   val = np.array([[-5.6312200e+00, -1.4374870e+01,  1.6280000e-02],
#          [-4.4962900e+00,  3.2185300e+00,  3.0897890e+00],
#          [ 3.6638700e+00,  6.2684000e-01, -3.2774100e+00],
#          [-4.7511100e+00,  7.4079600e+00, -3.4889400e+00],
#          [ 3.3489600e+00,  5.9988900e+00, -3.6005970e+00],
#          [ 1.1507200e+00,  1.6064490e+00,  2.8253800e+00],
#          [ 1.0708630e+01, -6.9044100e+00, -5.6815000e-01],
#          [ 2.6539800e+00, -7.1809000e-01,  7.1732000e-01],
#          [ 3.0786000e+00,  5.7680300e+00,  1.6818470e+00],
#          [ 3.1288000e+00, -4.6712000e-01,  7.0115000e-01],
#          [ 3.5025200e+00, -1.0283700e+00,  5.2879000e-01],
#          [ 1.7658300e+00,  3.3039700e+00, -2.8728000e-01],
#          [-2.8356000e-01, -7.8522000e+00,  1.4450730e+00],
#          [-6.3579000e-01, -1.1526860e+00,  2.6964800e+00],
#          [ 2.9685100e+00,  2.5052300e+00,  2.6058800e+00],
#          [-6.9799700e+00,  1.6087000e+00,  1.5982400e+00],
#          [ 2.2912200e+00, -7.7727500e+00, -2.1298750e+00],
#          [-3.4358100e+00, -4.8456100e+00,  2.8741400e+00],
#          [ 4.7602500e+00, -1.1259520e+01, -1.2851220e+00],
#          [ 5.2380700e+00, -2.9607600e+00,  9.9202000e-01],
#          [-2.6441700e+00, -2.8183400e+00, -7.4238000e-01],
#          [ 1.1961520e+01, -6.4593600e+00,  2.3016480e+00],
#          [-1.5829000e+00, -1.3143860e+00,  6.4467000e-01],
#          [ 5.1337000e+00,  8.5805000e-01,  4.1441850e+00],
#          [-3.0707360e+00,  2.9970100e+00,  4.1455000e-01],
#          [-6.0000000e-01,  8.0000000e-01,  1.0000000e+00],
#          [ 1.5090200e+00,  7.1584000e-01,  1.9848300e+00],
#          [-3.2795000e+00,  7.6784000e-01,  1.0351710e+00],
#          [ 1.5021900e+00, -1.8750000e-02,  2.8006000e-01],
#          [ 4.5894900e+00,  4.8452000e-01,  4.7912000e-01],
#          [-1.1943300e+00,  9.5822000e-02,  2.4036200e+00],
#          [ 7.0353600e+00, -4.9175000e-01,  2.4494000e-01],
#          [ 7.2897000e+00,  3.7622800e+00,  3.9012500e+00],
#          [ 3.4675200e+00, -5.3283600e+00, -1.4576000e+00],
#          [-3.2938500e+00,  3.0451500e+00, -6.0225000e-01],
#          [ 6.8859500e+00, -2.1454300e+00,  3.0504390e+00],
#          [-5.0282400e+00, -1.1003300e+00,  1.2250000e-02],
#          [ 1.7568630e+01,  6.1316000e-01, -2.2361000e-01],
#          [-2.6051100e+00,  2.1524200e+00, -2.7640190e+00],
#          [ 4.4354900e+00,  4.5775600e+00, -2.2185800e+00],
#          [-1.2598400e+00, -4.0537670e+00,  2.2279500e+00],
#          [ 6.4768800e+00, -1.4296400e+00,  2.4439810e+00],
#          [-1.6868300e+00, -1.6468840e+00, -1.1959000e-01],
#          [ 8.9545800e+00,  1.2452400e+00,  2.1364100e+00],
#          [ 3.2358720e+00, -2.3865500e+00, -2.0988800e+00],
#          [-6.0032400e+00,  4.4855600e+00, -8.4897000e-01],
#          [-4.5046600e+00, -6.8280000e-02, -6.8326000e-01],
#          [-2.5206000e-01,  2.7264300e+00, -3.3373500e+00],
#          [ 5.0235100e+00, -3.5933700e+00,  1.6142800e+00],
#          [-6.7073000e-01,  5.7539100e+00, -7.9485000e-01],
#          [ 1.0768810e+01,  3.7861120e+00,  1.0761000e+00],
#          [ 2.0000000e-01, -2.0000000e-01, -1.0000000e+00],
#          [ 3.9968000e+00,  8.2158000e-01,  3.5451100e+00],
#          [ 2.1438400e+00, -2.8696000e-01,  9.0714000e-01],
#          [-1.3118000e-01, -8.8957000e-01, -4.8370000e-01],
#          [ 2.3010100e+00,  9.0091000e-01,  2.8698900e+00],
#          [ 2.7504300e+00, -1.5753700e+00,  3.9637800e+00],
#          [-4.6864800e+00,  7.6078000e-01, -4.1601330e+00],
#          [ 4.0160000e+00, -4.8050000e-02,  6.9237700e-02],
#          [-3.2095200e+00, -3.1319700e+00, -4.2704000e-01],
#          [-3.5605000e+00, -5.4900000e-03, -3.6426060e+00],
#          [-1.4408600e+00, -5.2370000e-01, -1.5569300e+00],
#          [-1.6134000e-01, -1.1362300e+00, -8.1457000e-01],
#          [ 6.3853100e+00, -2.5592900e+00, -5.2862000e-01],
#          [ 7.0274000e-01, -1.1477000e+00,  2.3141100e+00],
#          [ 2.7762600e+00,  6.1881000e-01,  7.3026000e-01],
#          [ 1.1889610e+01,  4.2890300e+00,  1.4120000e-02],
#          [-3.5031400e+00, -2.6968800e+00, -1.7333330e+00],
#          [ 1.4845300e+00, -4.5396900e+00,  1.2794100e+00],
#          [-2.7014600e+00, -1.2621000e-01, -5.6688000e-01],
#          [-3.1526700e+00, -5.7308000e-01, -1.5480500e+00],
#          [ 2.8852800e+00,  1.6213660e+00, -1.8277000e-01],
#          [ 1.5507700e+00, -4.4542000e-01,  6.8424000e-01],
#          [-1.4898600e+00,  2.1014000e-01, -9.9136000e-01],
#          [-2.6686300e+00, -3.7280000e-01,  1.1083000e-01],
#          [ 1.1355000e-01, -2.5136500e+00, -2.9169960e+00],
#          [ 1.3851200e+00, -1.1825000e-01, -2.0747600e+00],
#          [ 3.3669600e+00, -2.3527600e+00,  5.3043000e-01],
#          [ 2.0733500e+00, -2.8623500e+00, -1.3401610e+00],
#          [-1.4496800e+00,  1.3372500e+00, -4.3247000e-01],
#          [ 5.9929600e+00,  3.9008100e+00, -3.8830920e+00],
#          [-3.3461000e+00, -4.4254000e-01, -1.2243700e+00],
#          [-1.4891000e-01,  1.0872500e+00, -1.4695600e+00],
#          [-1.4805100e+00,  1.7562800e+00,  5.9302000e-01],
#          [-1.8243200e+00,  2.3205000e-01, -5.3951000e-01],
#          [ 1.3213400e+00, -2.3494390e+00,  3.6692600e+00],
#          [-1.4743700e+00, -5.2370000e-02,  1.3672300e+00],
#          [ 6.9954800e+00, -4.7216400e+00, -4.1838300e+00],
#          [ 9.6070000e-01, -3.2238500e+00, -3.0976800e+00],
#          [-6.2105700e+00, -4.3828800e+00, -4.3862000e-01],
#          [ 6.0378900e+00, -8.8257200e+00, -2.8941000e-01],
#          [ 4.3589500e+00,  3.3830000e-01,  2.9664910e+00],
#          [ 3.1982100e+00,  5.4476000e-01, -4.3141000e-01],
#          [ 1.4223100e+00, -1.8663400e+00, -4.8651000e-01],
#          [-8.5123900e+00, -4.6233600e+00, -2.4546000e+00],
#          [ 8.0581300e+00,  1.1925520e+00, -2.7967000e-01],
#          [ 2.9921600e+00,  2.0061900e+00,  4.4725000e-01],
#          [ 2.7852500e+00, -5.4138000e-01,  3.0002610e+00],
#          [ 8.0000000e-01,  2.0000000e-01,  1.0000000e+00],
#          [-3.0465200e+00, -2.3059700e+00, -2.6919680e+00],
#          [ 1.8367000e+01,  8.0758000e+00,  9.4981300e-02],
#          [ 1.1428600e+00, -1.0708330e+01, -3.1901200e+00],
#          [-2.2172800e+00,  1.7844180e+00,  1.0224900e+00],
#          [-2.6120900e+00, -1.6995000e-01, -1.7841400e+00],
#          [ 1.5919700e+00, -1.2659000e+00, -1.0471800e+00],
#          [ 3.7989600e+00,  4.9532000e-01,  9.8113000e-01],
#          [ 1.7141300e+00,  1.0285700e+00, -1.0908000e-01],
#          [-2.8961900e+00,  2.8035000e-01, -4.5430000e-02],
#          [ 4.1238670e+00, -3.1745400e+00,  9.2271000e-01],
#          [-9.1289000e-01, -1.8403000e+00, -2.7221500e+00],
#          [-3.7877100e+00,  2.0770300e+00,  4.1603300e-01],
#          [-7.2925000e-01, -3.8724600e+00,  6.8782000e-01],
#          [ 6.7222900e+00, -2.4601000e+00,  2.7944200e+00],
#          [-1.1968100e+00,  2.5230000e-01, -3.9708000e-01],
#          [-3.2100000e-02,  4.7243300e+00,  1.1878240e+00],
#          [ 1.3540000e-01, -2.2571600e+00,  2.2686000e-01],
#          [-2.4509490e+00, -1.8113700e+00,  5.5776000e-01],
#          [ 4.6693200e+00,  8.4815000e-01, -1.0213920e+00],
#          [ 7.2040600e+00, -8.0503800e+00, -1.1830000e+00],
#          [ 4.5312300e+00,  2.0953500e+00,  1.5460200e+00],
#          [-1.7076300e+00, -3.7530480e+00, -5.5561000e-01],
#          [-3.8731600e+00, -3.8649200e+00, -2.9165397e+00],
#          [ 2.2315200e+00,  5.4663000e-01,  5.5834000e-01],
#          [ 1.4859800e+00, -3.0604100e+00,  6.6708000e-01],
#          [ 4.5824200e+00,  2.3719000e-01, -1.8880300e+00],
#          [ 9.1613000e-01, -2.8360700e+00, -1.0751500e+00],
#          [-2.9592300e+00, -3.2892800e+00, -2.2845100e+00],
#          [ 3.1790300e+00, -2.3992000e+00, -5.9056000e-01],
#          [-6.5231000e-01,  2.9163400e+00, -1.6050000e-01],
#          [-3.2349900e+00, -3.3948500e+00, -2.1186740e+00],
#          [-4.0988700e+00,  1.0211450e+01,  2.5941200e+00],
#          [ 1.4763250e+01, -7.1448000e-01,  2.3439100e+00],
#          [-2.8411400e+00,  1.2703200e+00,  8.2887000e-01],
#          [ 2.9642700e+00, -1.1648000e+00, -2.1620000e-01],
#          [ 3.9964100e+00,  2.6508000e-01,  5.1617000e-01],
#          [ 1.0096900e+00, -3.5138900e+00,  3.7244840e+00],
#          [ 1.5967400e+00,  1.5370100e+00,  2.1434900e+00],
#          [ 3.6316700e+00, -5.0505040e+00,  8.5989000e-01],
#          [ 3.3555500e-01,  1.3377200e+00,  1.2447000e-01],
#          [ 8.1935750e+00, -7.2415100e+00,  2.3709050e+00],
#          [-4.1304100e+00,  1.5371700e+00, -1.2304880e+00],
#          [ 2.5849500e+00,  7.1305000e-01,  3.2962700e+00],
#          [ 4.4612700e+00,  5.3438000e+00,  1.3250400e+00],
#          [-2.8202400e+00, -1.5720000e+00, -3.6557000e+00],
#          [ 5.4541000e-01,  9.3182000e-01, -1.7458500e+00],
#          [ 7.1155600e+00,  8.7680000e-01,  3.8895120e+00],
#          [-1.1980700e+00, -2.7926900e+00,  1.3176000e-01],
#          [ 6.0493900e+00, -3.7895500e+00, -1.4708510e+00],
#          [ 4.0040100e+00,  1.6978000e-01,  1.3539700e+00],
#          [-1.5725000e+00,  1.6627000e-01,  4.3909000e-01],
#          [ 1.4816200e+00,  8.9464000e-01, -1.5636000e-01],
#          [-9.9248000e-01,  2.3166200e+00,  1.1420000e-01],
#          [-5.0008700e+00,  2.0410000e-02, -9.1930600e-01],
#          [ 5.4710800e+00, -6.1218300e+00,  1.1775800e+00],
#          [-2.7065800e+00, -4.5024000e+00,  2.3538000e-01]])
#
#   theta, phi = 112.09872434893208, 359.352425425575
#
#   test1 = []
#   for value in val:
#     test1.append(angle(value, theta, phi))
#
#   test2 = np.transpose(angle2(val, theta, phi, "a", 0, 0))
#
#   for ite in range(len(test1)):
#     print(test1[ite])
#     print(test2[ite])
#     print("========================")
