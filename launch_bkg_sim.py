# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import subprocess
import os
import numpy as np
import multiprocessing as mp
import argparse
# Developped modules imports


def read_par(parfile):
  with open(parfile, "r") as f:
    lines = f.read().split("\n")
  geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes = None, None, None, None, None, None, None, None
  for line in lines:
    if line.startswith("@geometry"):
      geom = line.split(" ")[1]
    elif line.startswith("@revancfgfile"):
      revanf = line.split(" ")[1]
    elif line.startswith("@mimrecfile"):
      mimrecf = line.split(" ")[1]
    elif line.startswith("@cosimasourcefile"):
      source_base = line.split(" ")[1]
    elif line.startswith("@spectrafilepath"):
      spectra = line.split(" ")[1]
    elif line.startswith("@simtime"):
      simtime = float(line.split(" ")[1])
    elif line.startswith("@altitudes"):
      altitudes = map(float, line.split(" ")[1:])
    elif line.startswith("@latitudes"):
      latitudes = map(float, line.split(" ")[1:])
      latitudes = np.linspace(latitudes[0], latitudes[1], latitudes[2])
  return geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes


def make_directories(geometry, spectra):
  # Creating the bkg_source_spectra repertory if it doesn't exist
  if spectra.endswith("/"):
    directory = spectra.split("/")[-2]
  else:
    directory = spectra.split("/")[-1]
  if not directory in os.listdir("./bkg"):
    os.mkdir(spectra)
  # Creating a directory specific to the geometry
  geom_name = geometry.split(".geo.setup")[0].split("/")[-1]
  if not f"sim_{geom_name}" in os.listdir("./bkg"):
    os.mkdir(f"./bkg/sim_{geom_name}")
    # Creating the sim and rawsim repertories if they don't exist
  if not f"sim" in os.listdir(f"./bkg/sim_{geom_name}"):
    os.mkdir(f"./bkg/sim_{geom_name}/sim")
  if not f"rawsim" in os.listdir(f"./bkg/sim_{geom_name}"):
    os.mkdir(f"./bkg/sim_{geom_name}/rawsim")


def make_spectra(spectra, alt, lat):
  """

  """
  if not f"source-dat--alt_{alt}--lat_{lat}" in os.listdir(spectra):
    os.mkdir(f"{spectra}/source-dat--alt_{alt}--lat_{lat}")
  os.chdir("./bkg")
  subprocess.call(f"python CreateBackgroundSpectrumMEGAlib.py -i {lat} -a {alt}", shell=True)
  source_spectra = subprocess.getoutput("ls *Spec*.dat").split("\n")
  os.chdir("..")
  for spectrum in source_spectra:
    subprocess.call(f"mv ./bkg/{spectrum} {spectra}/source-dat--alt_{alt}--lat_{lat}", shell=True)


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


def make_tmp_source(alt, lat, geom, source_model, spectra):
  """

  """
  fname = f"tmp_{os.getpid()}.source"
  geom_name = geometry.split(".geo.setup")[0].split("/")[-1]
  sname = f"./bkg/sim_{geom_name}/sim/bkg_{alt}_{lat}_3600s"
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
          f.write(f"{source}.Beam FarFieldFileZenithDependent ./bkg/AlbedoPhotonBeam.dat")
        else:
          f.write(line)
      elif line.startswith(f"{source}.Spectrum"):
        particle_dat = f"{spectra}/source-dat--alt_{alt}--lat_{lat}/{particle}_Spec_{alt}km_{lat}deg.dat"
        f.write(f"{source}.Spectrum File {particle_dat}")
      elif line.startswith(f"{source}.Flux"):
        flux = read_flux_from_spectrum(particle_dat)
        f.write(f"{source}.Flux {flux}")
      else:
        f.write(line)
      f.write("\n")
  return fname, sname


def make_parameters(alts, lats, geomfile, source_model, rcffile, mimfile, spectra):
  """

  """
  parameters = []
  for alt in alts:
    for lat in lats:
      parameters.append((alt, lat, geomfile, source_model, rcffile, mimfile, spectra))
  return parameters


def run_bkg(params):
  # Making the different sources spectra :
  make_spectra(params[6], params[0], params[1])
  # Making a temporary source file using a source_model
  sourcefile, simname = make_tmp_source(params[0], params[1], params[2], params[3], params[6])
  # Making a generic name for files
  simfile, trafile = f"{simname}.inc1.id1.sim.gz", f"{simname}.inc1.id1.tra.gz"
  mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  mv_simfile, mv_trafile = f"{mv_simname}.inc1.id1.sim.gz", f"{mv_simname}.inc1.id1.tra.gz"
  #   Running the different simulations
  # Running cosima
  # subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'))
  # Running revan
  # subprocess.call(f"revan -g {params[2]} -c {params[3]} -f {simfile} -n -a; rm -f {simfile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  subprocess.call(f"revan -g {params[2]} -c {params[4]} -f {simfile} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  subprocess.call(f"mv {simfile} {mv_simfile}", shell=True)
  # Running mimrec
  subprocess.call(f"mimrec -g {params[2]} -c {params[5]} -f {trafile} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  subprocess.call(f"mv {trafile} {mv_trafile}", shell=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Multi-threaded automated MEGAlib runner. Parse a parameter file (mono-threaded) to generate commands that are executed by cosima and revan in a multi-threaded way.")
  parser.add_argument("-f", "--parameterfile", help="Path to parameter file used to generate commands")
  # parser.add_argument("-nc", "--nocosima", help="Does not run cosima", action="store_true")
  # parser.add_argument("-nr", "--norevan", help="Does not run revan", action="store_true")
  # parser.add_argument("-nm", "--nomimrec", help="Does not run mimrec", action="store_true")
  args = parser.parse_args()
  if args.parameterfile:
    print(f"Running of {args.parameterfile} parameter file")
    geometry, revanfile, mimrecfile, source_base, spectra, simtime, latitudes, altitudes = read_par(args.parameterfile)

    parameters = make_parameters(altitudes, latitudes, geometry, source_base, revanfile, mimrecfile, spectra)
    print(f"{len(parameters)} Commands have been parsed")
    with mp.Pool() as pool:
      pool.map(run_bkg, parameters)
  else:
    print("Missing parameter file or geometry - not running.")

