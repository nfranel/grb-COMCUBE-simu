# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# file to launch background simulations

# Package imports
import subprocess
import glob
import os
import multiprocessing as mp
import argparse

# Developped modules imports
from src.General.funcmod import read_bkgpar


def make_directories(geomfile, spectrapath):
  """
  Create the directories in which the simulations are saved
  :param geomfile: geometry used for the simulations
  :param spectrapath: path of the folder in which the background spectra are saved
  """
  # Creating the bkg_source_spectra repertory if it doesn't exist
  if not spectrapath.split("/")[-1] in os.listdir("../Data/bkg"):
    os.mkdir(spectrapath)
  # Creating a directory specific to the geometry
  geom_name = geomfile.split(".geo.setup")[0].split("/")[-1]
  if f"sim_{geom_name}" not in os.listdir("../Data/bkg"):
    os.mkdir(f"../Data/bkg/sim_{geom_name}")
    # Creating the sim and rawsim repertories if they don't exist
  if f"sim" not in os.listdir(f"../Data/bkg/sim_{geom_name}"):
    os.mkdir(f"../Data/bkg/sim_{geom_name}/sim")
  if f"rawsim" not in os.listdir(f"../Data/bkg/sim_{geom_name}"):
    os.mkdir(f"../Data/bkg/sim_{geom_name}/rawsim")


def make_spectra(params):
  """
  Create background spectra of different particles for different altitudes and latitudes
  :param params: parameters from the parameter file
  """
  spectrapath, alt, lat = params[6], params[0], params[1]
  bkg_code = "./src/Background"
  if f"source-dat--alt_{alt:.1f}--lat_{lat:.1f}" not in os.listdir(spectrapath):
    os.mkdir(f"{spectrapath}/source-dat--alt_{alt:.1f}--lat_{lat:.1f}")
  os.chdir(bkg_code)
  subprocess.call(f"python CreateBackgroundSpectrumMEGAlib.py -i {lat} -a {alt}", shell=True)
  # source_spectra = subprocess.getoutput(f"ls *_Spec_{alt:.1f}km_{lat:.1f}deg.dat").split("\n")
  source_spectra = glob.glob(f"*_Spec_{alt:.1f}km_{lat:.1f}deg.dat")
  os.chdir("../../")
  for spectrum in source_spectra:
    subprocess.call(f"mv {bkg_code}/{spectrum} {spectrapath}/source-dat--alt_{alt:.1f}--lat_{lat:.1f}", shell=True)


def read_flux_from_spectrum(file):
  """
  Reads the flux writen in the spectrum file and returns the value
  :param file: spectrum file
  :returns: Flux [/cm^2/s]
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


def make_tmp_source(alt, lat, geom, source_model, spectrapath, simduration):
  """
  Creates a temporary source file based on a model "source model"
  :param alt: altitude for the background simulation
  :param lat: latitude for the background simulation
  :param geom: geometry used for the background simulation
  :param source_model: model used to create temporary source files
  :param spectrapath: path to spectra folder
  :param simduration: duration of the background simulation
  :returns: name of the temporary source file, name of the simulation without the extension
  """
  fname = f"tmp_{os.getpid()}.source"
  geom_name = geom.split(".geo.setup")[0].split("/")[-1]
  sname = f"../Data/bkg/sim_{geom_name}/sim/bkg_{alt:.1f}_{lat:.1f}_{simduration:.0f}s"
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
          print("Name of run is not valid. Check parameter file and use Bckgrnd.")
      elif line.startswith(f"{run}.FileName"):
        f.write(f"{run}.FileName {sname}")
      elif line.startswith(f"{run}.Time"):
        f.write(f"{run}.Time {simduration}")
      elif line.startswith(f"{run}.Source"):
        source = line.split(" ")[-1]
        particle = source.split("Source")[0]
        if particle in source_list:
          f.write(line)
        else:
          print("Name of source is not valid. Should be one of the ones for which a spectrum was calculated.")
      elif line.startswith(f"{source}.Beam"):
        if particle in ["AtmosphericNeutrons", "AlbedoPhotons"]:
          f.write(f"{source}.Beam FarFieldFileZenithDependent ../Data/bkg/AlbedoPhotonBeam.dat")
        else:
          f.write(line)
      elif line.startswith(f"{source}.Spectrum"):
        particle_dat = f"{spectrapath}/source-dat--alt_{alt:.1f}--lat_{lat:.1f}/{particle}_Spec_{alt:.1f}km_{lat:.1f}deg.dat"
        f.write(f"{source}.Spectrum File {particle_dat}")
      elif line.startswith(f"{source}.Flux"):
        flux = read_flux_from_spectrum(particle_dat)
        f.write(f"{source}.Flux {flux}")
      else:
        f.write(line)
      f.write("\n")
  return fname, sname


def make_parameters(alts, lats, geomfile, source_model, rcffile, mimfile, spectrapath, simduration):
  """
  Creates a lists of parameters for several altitudes and latitudes
  :param alts: altitudes for the background simulation
  :param lats: latitudes for the background simulation
  :param geomfile: geometry used for the background simulation
  :param source_model: model used to create temporary source files
  :param rcffile: revan configuration file to treat raw simulations
  :param mimfile: mimrec configuration file to extract simulations treated with revan
  :param spectrapath: path to spectra folder
  :param simduration: duration of the background simulation
  """
  parameters_container = []
  for alt in alts:
    for lat in lats:
      parameters_container.append((alt, lat, geomfile, source_model, rcffile, mimfile, spectrapath, simduration))
  return parameters_container


def run(command, error_file, expected_file):
  """
  Runs a command
  :param command: str, shell command to run
  :param error_file: str, name of the error logfile
  """
  proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  folder = f"{expected_file.split('/sim/')[0]}/sim/"
  simname = expected_file.split("/sim/")[-1]
  if proc.stderr != "":
    with open(error_file, "a") as errfile:
      errormess = "\n=========================================================================================================\n" + f"ERROROUTPUT : {simname}\n" + proc.stderr + "\n"
      errfile.write(errormess)
  if not (simname in os.listdir(folder)):
    with open(error_file, "a") as errfile:
      errormess = "\n=========================================================================================================\n" + f"NOFILE output : {simname}\n" + proc.stdout + "\n"
      errfile.write(errormess)


def run_bkg(params):
  """
  Runs the cosima, revan and mimrec programs and either move to rawsim or remove the .sim.gz and .tra.gz files
  :param params: list of parameters to run the simulation
  """
  # Making a temporary source file using a source_model
  sourcefile, simname = make_tmp_source(params[0], params[1], params[2], params[3], params[6], params[7])
  # Making a generic name for files
  simfile, trafile, extrfile = f"{simname}.inc1.id1.sim.gz", f"{simname}.inc1.id1.tra.gz", f"{simname}.inc1.id1.extracted.tra"
  # mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  # mv_simfile, mv_trafile = f"{mv_simname}.inc1.id1.sim.gz", f"{mv_simname}.inc1.id1.tra.gz"

  #   Running the different simulations
  print(f"Running bkg simulation : {simname}")
  # Running cosima
  # subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  # subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'))
  run(f"cosima -z {sourcefile}; rm -f {sourcefile}", f"{simname.split('/sim/')[0]}/cosima_errlog.txt", simfile)

  # Running revan
  # subprocess.call(f"revan -g {params[2]} -c {params[3]} -f {simfile} -n -a; rm -f {simfile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  # subprocess.call(f"revan -g {params[2]} -c {params[4]} -f {simfile} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  run(f"revan -g {params[2]} -c {params[4]} -f {simfile} -n -a", f"{simname.split('/sim/')[0]}/revan_errlog.txt", trafile)
  # Moving the cosima file in rawsim or removing it
  # subprocess.call(f"mv {simfile} {mv_simfile}", shell=True)
  subprocess.call(f"rm -f {simfile}", shell=True)

  # Running mimrec
  # subprocess.call(f"mimrec -g {params[2]} -c {params[5]} -f {trafile} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  run(f"mimrec -g {params[2]} -c {params[5]} -f {trafile} -x -n", f"{simname.split('/sim/')[0]}/mimrec_errlog.txt", extrfile)
  # Moving the revan analyzed file in rawsim or removing it
  # subprocess.call(f"mv {trafile} {mv_trafile}", shell=True)
  subprocess.call(f"rm -f {trafile}", shell=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Multi-threaded automated MEGAlib runner. Parse a parameter file (mono-threaded) to generate commands that are executed by cosima and revan in a multi-threaded way.")
  parser.add_argument("-f", "--parameterfile", help="Path to parameter file used to generate commands")
  args = parser.parse_args()
  if args.parameterfile:
    # Reading the param file
    print(f"Running of {args.parameterfile} parameter file")
    geometry, revanfile, mimrecfile, source_base, spectra, simtime, latitudes, altitudes = read_bkgpar(args.parameterfile)
    # Creating the required directories
    make_directories(geometry, spectra)
    # Creating the parameter list
    parameters = make_parameters(altitudes, latitudes, geometry, source_base, revanfile, mimrecfile, spectra, simtime)
    print("===================================================================")
    print(f"{len(parameters)} Commands have been parsed")
    print("===================================================================")

    print("===================================================================")
    print("Running the creation of spectra")
    print("===================================================================")
    with mp.Pool() as pool:
      pool.map(make_spectra, parameters)
    print("===================================================================")
    print("Running the background simulations and extraction")
    print("===================================================================")
    with mp.Pool() as pool:
      pool.map(run_bkg, parameters)
  else:
    print("Missing parameter file or geometry - not running.")
