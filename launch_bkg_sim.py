# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import subprocess
import os
import numpy as np
import multiprocessing as mp
# Developped modules imports


def make_spectra(alt, lat):
  """

  """
  if not f"source-dat--alt_{alt}--lat_{lat}" in os.listdir("./bkg_source_spectra"):
    os.mkdir(f"./bkg_source_spectra/source-dat--alt_{alt}--lat_{lat}")
  subprocess.call(f"python CreateBackgroundSpectrumMEGAlib.py -i {lat} -a {alt}", shell=True)
  source_spectra = subprocess.getoutput("ls *Spec*.dat").split("\n")
  for spectrum in source_spectra:
    subprocess.call(f"mv {spectrum} ./bkg_source_spectra/source-dat--alt_{alt}--lat_{lat}", shell=True)


def read_flux_from_spectrum(file):
  """

  """
  with open(file, "r") as f:
    lines = f.read().split("\n")
  line_ite = 0
  line = lines[line_ite]
  while line.startswith("#") and line_ite <= 10:
    if line.startswith("# Integral Flux:"):
      return float(line.split("# Integral Flux:")[1].split("#")[0].strip())
    line_ite += 1
    line = lines[line_ite]


def make_tmp_source(alt, lat, geom, source_model):
  """

  """
  fname = f"tmp_{os.getpid()}.source"
  sname = f"./sim/bkg_{alt}_{lat}_3600s"
  source_list = ["SecondaryElectrons", "AtmosphericNeutrons", "AlbedoPhotons", "SecondaryPositrons", "SecondaryProtonsUpward", "SecondaryProtonsDownward", "PrimaryElectrons", "CosmicPhotons", "PrimaryPositrons", "PrimaryProtons"]
  with open(source_model) as f:
    lines = f.read().split("\n")
  with open(fname, "w") as f:
    run, source = "", ""  # "GRBsource" or "GRBsourcenp"
    for line in lines:
      if line.startswith("Geometry"):
        f.write(f"Geometry {geom}")
      elif line.startswith("Run"):
        run = line.split(" ")[-1]
        if run == "Bckgrnd":
          f.write(line)
        else:
          print("Name of run is not valid. Check parameter file and use either GRBpol for polarized run or GRBnpol for unpolarized run.")
      elif line.startswith(f"{run}.FileName"):
        f.write(f"{run}.FileName {sname}")
      elif line.startswith(f"{run}.Time"):
        f.write(f"{run}.Time {3600}")
      elif line.startswith(f"{run}.Source"):
        source = line.split(" ")[-1]
        particle = source.split("Source")[0]
        if particle in source_list:
          f.write(line)
        else:
          print("Name of source is not valid. Should be one of the ones for which a spectrum was calculated.")
      elif line.startswith(f"{source}.Beam"):
        if particle in ["AtmosphericNeutrons", "AlbedoPhotons"]:
          f.write(f"{source}.Beam FarFieldFileZenithDependent ./AlbedoPhotonBeam.dat")
        else:
          f.write(line)
      elif line.startswith(f"{source}.Spectrum"):
        particle_dat = f"./bkg_source_spectra/source-dat--alt_{alt}--lat_{lat}/{particle}_Spec_{alt}km_{lat}deg.dat"
        f.write(f"{source}.Spectrum File {particle_dat}")
      elif line.startswith("{source}.Flux"):
        flux = read_flux_from_spectrum(particle_dat)
        f.write(f"{source}.Flux {flux}")
      else:
        f.write(line)
      f.write("\n")
  return fname, sname


def make_parameters(alts, lats, geomfile, source_model, rcffile, mimfile):
  """

  """
  parameters = []
  for alt in alts:
    for lat in lats:
      parameters.append((alt, lat, geomfile, source_model, rcffile, mimfile))
  return parameters


def run_bkg(params):
  # Making the different sources spectra :
  make_spectra(params[0], params[1])
  # Making a temporary source file using a source_model
  sourcefile, simname = make_tmp_source(params[0], params[1], params[2], params[3])
  # Making a generic name for files
  simfile, trafile = f"{simname}.inc1.id1.sim.gz", f"{simname}.inc1.id1.tra.gz"
  mv_simname = f"{simname.split('/')[0]}rawsim{simname.split('/')[1]}"
  mv_simfile, mv_trafile = f"{simname}.inc1.id1.sim.gz", f"{simname}.inc1.id1.tra.gz"
  #   Running the different simulations
  # Running cosima
  print(f"cosima -z {sourcefile}; rm {sourcefile}")
  subprocess.call(f"ls", shell=True)
  subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  # Running revan
  # subprocess.call(f"revan -g {params[2]} -c {params[3]} -f {simfile} -n -a; rm -f {simfile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  # subprocess.call(f"revan -g {params[2]} -c {params[3]} -f {simfile} -n -a", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  # subprocess.call(f"mv {simfile} {mv_simfile}", shell=True)
  # Running mimrec
  # subprocess.call(f"mimrec -g {params[2]} -c {params[4]} -f {trafile} -x -n", shell=True)
  # subprocess.call(f"mv {trafile} {mv_trafile}", shell=True)


geometry = "../geom/COMCUBE_v134.geo.setup"
source_base = "./bkgrnd_model.source"
revanfile = "../cfgs/revanv1.cfg"
mimrecfile ="../cfgs/mimrec10-1000single.cfg"
# Creating the different source spectra for different latitudes
# latitudes = np.linspace(-90, 90, 3)
# altitudes = [400, 500]
latitudes = [0]
altitudes = [400]
# repertory for bkg source spectra : source-dat--alt_{altitude}--lat_{latitudes}
# source spectra names : {Particle}_Spec_{altitude}km_{latitude}deg.dat

os.chdir("./bkg/")
# Creating the bkg_source_spectra repertory if it doesn't exist
if not "bkg_source_spectra" in os.listdir("."):
  os.mkdir("bkg_source_spectra")
# Creating the sim and rawsim repertories if they don't exist
geom_name = geometry.split(".geo.setup")[0].split("/")[-1]
if not f"sim--{geom_name}" in os.listdir("."):
  os.mkdir(f"sim--{geom_name}")
if not f"rawsim--{geom_name}" in os.listdir("."):
  os.mkdir(f"rawsim--{geom_name}")


parameters = make_parameters(altitudes, latitudes, geometry, source_base, revanfile, mimrecfile)

with mp.Pool() as pool:
  pool.map(run_bkg, parameters)


# bkg_{alt}_{lat}_3600s.inc1.id1.sim.gz
# bkg_{alt}_{lat}_3600s.inc1.id1.tra.gz
# bkg_{alt}_{lat}_3600s.inc1.id1.extracted.tra
