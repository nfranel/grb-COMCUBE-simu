# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  launch_bkg_sim.py
# Contains various functions and a main to run automatically the background simulations
# ================================================================

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
  Creates the directory tree required to store background simulation outputs,
  including the spectra folder, the folder containing the simulations for a given geometry, and
  its sim/ and rawsim/ subdirectories. Directories that already exist are
  left untouched.
  :param geomfile: str, path to the geometry file (must end with ".geo.setup");
      the geometry name is extracted from the filename stem
  :param spectrapath: str, path of the folder in which the background spectra
      are saved; created inside ../Data/bkg/ if it does not exist
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
  Generates background particle spectra for a single (altitude, latitude) point
  by calling the external CreateBackgroundSpectrumMEGAlib.py script, then moves
  the resulting .dat files into the appropriate subfolder of spectrapath.
  :param params: tuple, parameter tuple as produced by make_parameters():
      params[0] - float, altitude for the background simulation [km]
      params[1] - float, geomagnetic latitude for the background simulation [deg]
      params[6] - str, path to the spectra folder
  """
  spectrapath, alt, lat = params[6], params[0], params[1]
  bkg_code = "./src/Background"
  if f"source-dat--alt_{alt:.1f}--lat_{lat:.1f}" not in os.listdir(spectrapath):
    os.mkdir(f"{spectrapath}/source-dat--alt_{alt:.1f}--lat_{lat:.1f}")
  os.chdir(bkg_code)
  subprocess.call(f"python CreateBackgroundSpectrumMEGAlib.py -i {lat} -a {alt}", shell=True)
  source_spectra = glob.glob(f"*_Spec_{alt:.1f}km_{lat:.1f}deg.dat")
  os.chdir("../../")
  for spectrum in source_spectra:
    subprocess.call(f"mv {bkg_code}/{spectrum} {spectrapath}/source-dat--alt_{alt:.1f}--lat_{lat:.1f}", shell=True)


def read_flux_from_spectrum(file):
  """
  Reads the integrated flux value written in the header of a background spectrum
  file and returns it as a float.
  :param file: str, path to the spectrum file; the flux is expected on a header
      line starting with "# Integral Flux:" within the first 10 lines
  :returns: float, integral flux read from the file [/cm^2/s]
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
  return None


def make_tmp_source(alt, lat, geom, source_model, spectrapath, simduration):
  """
  Creates a temporary cosima source file for a single background simulation point
  by filling in the geometry, run name, simulation duration, beam type, particle
  spectra, and fluxes for all background particle species into a template source file.
  :param alt: float, altitude for the background simulation [km]
  :param lat: float, geomagnetic latitude for the background simulation [deg]
  :param geom: str, path to the geometry file used for the simulation
  :param source_model: str, path to the template source file to use as a base
  :param spectrapath: str, path to the folder containing the particle spectrum files
  :param simduration: float, duration of the background simulation [s]
  :returns: str, str, path to the temporary source file created, base path and
      stem for the simulation output files (without extension)
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
          f.write(f"{source}.Beam FarFieldFileZenithDependent ./src/Background/AlbedoPhotonBeam.dat")
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
  Builds the full list of parameter tuples for all (altitude, latitude) simulation
  points, one tuple per point, ready to be mapped over by the multiprocessing pool.
  :param alts: list or np.ndarray, altitudes for the background simulation [km]
  :param lats: list or np.ndarray, geomagnetic latitudes for the background simulation [deg]
  :param geomfile: str, path to the geometry file used for the simulations
  :param source_model: str, path to the template source file
  :param rcffile: str, path to the revan configuration file
  :param mimfile: str, path to the mimrec configuration file
  :param spectrapath: str, path to the folder containing the particle spectrum files
  :param simduration: float, duration of each background simulation [s]
  :returns: list, list of tuples, each containing
      (alt, lat, geomfile, source_model, rcffile, mimfile, spectrapath, simduration)
  """
  parameters_container = []
  for alt in alts:
    for lat in lats:
      parameters_container.append((alt, lat, geomfile, source_model, rcffile, mimfile, spectrapath, simduration))
  return parameters_container


def autorun(command, error_file, expected_file):
  """
  Executes a shell command and logs any stderr output or missing output files
  to an error log file.
  :param command: str, shell command to run
  :param error_file: str, path to the error log file where stderr output and
      missing-file warnings are appended
  :param expected_file: str, path to the output file that the command is
      expected to produce; if absent after execution, a warning is logged
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
  Runs the full cosima -> revan -> mimrec pipeline for a single background
  simulation point (one altitude/latitude pair). The raw .sim.gz file is
  moved to rawsim/ after cosima, and the .tra.gz file is moved to rawsim/
  after mimrec extraction.
  :param params: tuple, parameter tuple as produced by make_parameters():
      params[0] - float, altitude [km]
      params[1] - float, geomagnetic latitude [deg]
      params[2] - str, path to the geometry file
      params[3] - str, path to the template source file
      params[4] - str, path to the revan configuration file
      params[5] - str, path to the mimrec configuration file
      params[6] - str, path to the spectra folder
      params[7] - float, simulation duration [s]
  """
  # Making a temporary source file using a source_model
  sourcefile, simname = make_tmp_source(params[0], params[1], params[2], params[3], params[6], params[7])
  # Making a generic name for files
  simfile, trafile, extrfile = f"{simname}.inc1.id1.sim.gz", f"{simname}.inc1.id1.tra.gz", f"{simname}.inc1.id1.extracted.tra"
  mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  mv_simfile, mv_trafile = f"{mv_simname}.inc1.id1.sim.gz", f"{mv_simname}.inc1.id1.tra.gz"

  #   Running the different simulations
  print(f"Running bkg simulation : {simname}")
  # Running cosima
  autorun(f"cosima -z {sourcefile}; rm -f {sourcefile}", f"{simname.split('/sim/')[0]}/cosima_errlog.txt", simfile)

  # Running revan
  autorun(f"revan -g {params[2]} -c {params[4]} -f {simfile} -n -a", f"{simname.split('/sim/')[0]}/revan_errlog.txt", trafile)
  # Moving the cosima file in rawsim or removing it
  subprocess.call(f"mv {simfile} {mv_simfile}", shell=True)
  # subprocess.call(f"rm -f {simfile}", shell=True)

  # Running mimrec
  autorun(f"mimrec -g {params[2]} -c {params[5]} -f {trafile} -x -n", f"{simname.split('/sim/')[0]}/mimrec_errlog.txt", extrfile)
  # Moving the revan analyzed file in rawsim or removing it
  subprocess.call(f"mv {trafile} {mv_trafile}", shell=True)
  # subprocess.call(f"rm -f {trafile}", shell=True)


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
