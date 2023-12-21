from MAllSourceData import AllSourceData
from time import time


init_time = time()
grb_sim_param = "/pdisk/ESA/test--400km--0-0-0--27sat/polGBM.par"
bkg_param = "./bkg/bkg-v134.par"
mu_param = "./mu100/mu100-v134.par"
erg = (30, 1000)
arm = 180
test = AllSourceData(grb_sim_param, bkg_param, mu_param, erg, arm, parallel="all")
test.make_const()
test.verif_const()
test.analyze()
test.verif_const()
# test.alldata[0][0][0].show()
test.count_triggers()
# test.source_information(0)
print("=======================================")
print("processing time : ", time()-init_time, "seconds")
print("=======================================")
test.spectral_information()

# cat = Catalog("./longGBM.txt", [4, '\n', 5, '|', 2000])

