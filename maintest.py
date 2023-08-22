from dataclass import *
from time import time


init_time = time()
bkg = "./backgrounds/bkg"  # _background_sat0_0000_90.0_0.0.inc1.id1.extracted.tra"
param = "./test/polGBM.par"
erg = (10, 1000)
arm = 180
test = AllSourceData(bkg, param, erg, arm, parallel=True)
test.make_const()
test.analyze()
print("=======================================")
print("processing time : ", time()-init_time, "seconds")
print("=======================================")

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
