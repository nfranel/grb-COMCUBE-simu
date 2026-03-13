# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  launch_mu100_sim.py
# Contains various functions and a main to run automatically the simulations for estimating instrument response (mu100 and Seff)
# ================================================================

# Package imports
import subprocess
import os
import numpy as np
import multiprocessing as mp
import argparse
# Developped modules imports
from src.General.funcmod import band, read_mupar, use_scipyquad


def make_directories(geomfile):
  """
  Creates the directory tree required to store mu100 simulation outputs for a
  given geometry, the folder containing the simulations for a given geometry and its sim/ and
  rawsim/ subdirectories. Directories that already exist are left untouched.
  :param geomfile: str, path to the geometry file (must end with ".geo.setup");
      the geometry name is extracted from the filename stem
  """
  # Creating a directory specific to the geometry
  geom_name = geomfile.split(".geo.setup")[0].split("/")[-1]
  if f"sim_{geom_name}" not in os.listdir("../Data/mu100"):
    os.mkdir(f"../Data/mu100/sim_{geom_name}")
    # Creating the sim and rawsim repertories if they don't exist
  if f"sim" not in os.listdir(f"../Data/mu100/sim_{geom_name}"):
    os.mkdir(f"../Data/mu100/sim_{geom_name}/sim")
  if f"rawsim" not in os.listdir(f"../Data/mu100/sim_{geom_name}"):
    os.mkdir(f"../Data/mu100/sim_{geom_name}/rawsim")


def make_spectrum(filepath, bandpar):
  """
  Creates a Band spectral file representing an average GRB spectrum, sampled
  on a log-spaced energy grid between 10 and 1000 keV. The file is written
  only if it does not already exist.
  :param filepath: str, path to the folder where the spectrum file is saved;
      the file is named Band_spectrum.dat inside this folder
  :param bandpar: list or tuple, Band function parameters in the order:
      [amplitude [ph/cm2/keV/s], alpha, beta, epeak [keV], epivot [keV]]
  """
  if not (f"{filepath}/Band_spectrum.dat" in os.listdir(filepath)):
    log_energy = np.logspace(1, 3, 100)  # energy (log scale)
    with open(f"{filepath}/Band_spectrum.dat", "w") as f:
      f.write("#model band:  ")
      f.write(f"ampl={bandpar[0]}ph/cm2/keV/s, alpha={bandpar[1]}, beta={bandpar[2]}, epeak={bandpar[3]}keV, epivot={bandpar[4]}keV\n")
      f.write("\nIP LOGLOG\n\n")
      for E in log_energy:
        f.write(f"DP {E} {band(E, bandpar[0], bandpar[1], bandpar[2], bandpar[3], bandpar[4])}\n")
      f.write("\nEN\n\n")


def make_tmp_source(dec, ra, geom, source_model, spectrapath, timepol, timeunpol, flux):
  """
  Creates a temporary cosima source file for a single mu100 simulation point
  by filling in the geometry, beam direction, spectrum, simulation durations,
  and flux into a template source file.
  :param dec: float, declination of the source in the satellite frame [deg]
  :param ra: float, right ascension of the source in the satellite frame [deg]
  :param geom: str, path to the geometry file used for the simulation
  :param source_model: str, path to the template source file to use as a base
  :param spectrapath: str, path to the folder containing the Band spectrum file
  :param timepol: float, duration of the polarized simulation [s]
  :param timeunpol: float, duration of the unpolarized simulation [s]
  :param flux: float, source flux used in the simulation [ph/cm2/s]
  :returns: str, str, path to the temporary source file created, base path and
      stem for the simulation output files (without pol/unpol suffix or extension)
  """
  fname = f"tmp_{os.getpid()}.source"
  geom_name = geometry.split(".geo.setup")[0].split("/")[-1]
  sname = f"../Data/mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}"
  with open(source_model) as f:
    lines = f.read().split("\n")
  with open(fname, "w") as f:
    run, source = "", ""  # "GRBsource" or "GRBsourcenp"
    for line in lines:
      if line.startswith("Geometry"):
        f.write(f"Geometry {geom}")
      elif line.startswith("Run"):
        run = line.split(" ")[-1]
        if run == "GRBpol" or run == "GRBnpol":
          f.write(line)
        else:
          print("Name of run is not valid. Check parameter file and use either GRBpol for polarized run or GRBnpol for unpolarized run.")
      elif line.startswith(f"{run}.FileName"):
        if run == "GRBpol":
          f.write(f"{run}.FileName {sname}pol")
        elif run == "GRBnpol":
          f.write(f"{run}.FileName {sname}unpol")
      elif line.startswith(f"{run}.Time"):
        if run == "GRBpol":
          f.write(f"{run}.Time {timepol}")
        elif run == "GRBnpol":
          f.write(f"{run}.Time {timeunpol}")
      elif line.startswith(f"{run}.Source"):
        source = line.split(" ")[-1]
        f.write(line)
      elif line.startswith(f"{source}.Beam") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Beam FarFieldPointSource {dec} {ra}")
      elif line.startswith(f"{source}.Spectrum") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Spectrum File {spectrapath}/Band_spectrum.dat")
      elif line.startswith(f"{source}.Flux") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Flux {flux}")
      else:
        f.write(line)
      f.write("\n")
  return fname, sname


def make_ra_list(ra_list, dec):
  """
  Builds a right ascension grid for a given declination such that the number
  of RA samples scales with sin(dec), giving a roughly uniform angular density
  on the sphere. The poles (dec = 0 or 180) receive only a single sample at RA = 0.
  :param ra_list: list, three-element list [ra_min, ra_max, n_ra_equator] where
      ra_min and ra_max are the RA bounds [deg] and n_ra_equator is the number
      of RA points at the equator
  :param dec: float, declination at which the RA grid is computed [deg]
  :returns: list or np.ndarray, right ascension values for this declination [deg]
  """
  if dec == 0 or dec == 180:
    new_ra = [0.0]
  else:
    new_ra = np.around(np.linspace(ra_list[0], ra_list[1], np.max([4, int(np.sin(np.deg2rad(dec)) * ra_list[2])]), endpoint=False), 1)
  return new_ra


def make_parameters(dec_list, ra_list, geomfile, source_model, spectrapath, timepol, timeunpol, flux, rcffile, mimfile):
  """
  Builds the full list of parameter tuples for all (dec, ra) simulation points,
  one tuple per point, ready to be mapped over by the multiprocessing pool.
  :param dec_list: list, three-element list [dec_min, dec_max, n_dec] defining
      the declination grid [deg]
  :param ra_list: list, three-element list [ra_min, ra_max, n_ra_equator] defining
      the right ascension grid [deg]
  :param geomfile: str, path to the geometry file used for the simulations
  :param source_model: str, path to the template source file
  :param spectrapath: str, path to the folder containing the Band spectrum file
  :param timepol: float, duration of the polarized simulation [s]
  :param timeunpol: float, duration of the unpolarized simulation [s]
  :param flux: float, source flux used in the simulations [ph/cm2/s]
  :param rcffile: str, path to the revan configuration file
  :param mimfile: str, path to the mimrec configuration file
  :returns: list, list of tuples, each containing
      (dec, ra, geomfile, source_model, spectrapath, timepol, timeunpol, flux, rcffile, mimfile)
  """
  parameters_container = []
  for dec in np.linspace(dec_list[0], dec_list[1], dec_list[2]):
    for ra in make_ra_list(ra_list, dec):
      parameters_container.append((dec, ra, geomfile, source_model, spectrapath, timepol, timeunpol, flux, rcffile, mimfile))
  return parameters_container


def autorun(command, error_file, expected_file, expected_file_unpol=None):
  """
  Executes a shell command and logs any stderr output or missing output files
  to an error log file.
  :param command: str, shell command to run
  :param error_file: str, path to the error log file where stderr output and
      missing-file warnings are appended
  :param expected_file: str, path to the output file that the command is
      expected to produce; if absent after execution, a warning is logged
  :param expected_file_unpol: str, path to the output file for non polarised
      simulations that the command is expected to produce. It is expected to be
      used to run with cosima only; if absent after execution, a warning is logged
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
  if expected_file_unpol is not None:
    simname2 = expected_file_unpol.split("/sim/")[-1]
    if not (simname2 in os.listdir(folder)):
      with open(error_file, "a") as errfile:
        errormess = "\n=========================================================================================================\n" + f"NOFILE output : {simname2}\n" + proc.stdout + "\n"
        errfile.write(errormess)


def run_mu(params):
  """
  Runs the full cosima -> revan -> mimrec pipeline for a single mu100 simulation
  point (one dec/ra pair), for both the polarized and unpolarized runs.
  The raw .sim.gz files are removed after revan processing, and the .tra.gz files
  are removed after mimrec extraction change the comments to move them instead.
  :param params: tuple, parameter tuple as produced by make_parameters():
      params[0] - float, declination [deg]
      params[1] - float, right ascension [deg]
      params[2] - str, path to the geometry file
      params[3] - str, path to the template source file
      params[4] - str, path to the spectra folder
      params[5] - float, polarized simulation duration [s]
      params[6] - float, unpolarized simulation duration [s]
      params[7] - float, source flux [ph/cm2/s]
      params[8] - str, path to the revan configuration file
      params[9] - str, path to the mimrec configuration file
  """
  # Making a temporary source file using a source_model
  sourcefile, simname = make_tmp_source(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
  # Making a generic name for files
  simfilepol, trafilepol, extrfilepol = f"{simname}pol.inc1.id1.sim.gz", f"{simname}pol.inc1.id1.tra.gz", f"{simname}pol.inc1.id1.extracted.tra"
  simfileunpol, trafileunpol, extrfileunpol = f"{simname}unpol.inc1.id1.sim.gz", f"{simname}unpol.inc1.id1.tra.gz", f"{simname}unpol.inc1.id1.extracted.tra"
  mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  mv_simfilepol, mv_trafilepol = f"{mv_simname}pol.inc1.id1.sim.gz", f"{mv_simname}pol.inc1.id1.tra.gz"
  mv_simfileunpol, mv_trafileunpol = f"{mv_simname}unpol.inc1.id1.sim.gz", f"{mv_simname}unpol.inc1.id1.tra.gz"

  #   Running the different simulations
  print(f"Running mu100 simulation : {simname}")
  # Running cosima
  # OLD VERSION (for debugging) subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'))
  autorun(f"cosima -z {sourcefile}; rm -f {sourcefile}", f"{simname.split('/sim/')[0]}/cosima_errlog.txt", simfilepol, expected_file_unpol=simfileunpol)

  # Running revan
  # OLD VERSION (for debugging) subprocess.call(f"revan -g {params[2]} -c {params[8]} -f {simfilepol} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  autorun(f"revan -g {params[2]} -c {params[8]} -f {simfilepol} -n -a", f"{simname.split('/sim/')[0]}/pol_revan_errlog.txt", trafilepol)
  # Moving the cosima pol file in rawsim or removing it
  # subprocess.call(f"mv {simfilepol} {mv_simfilepol}", shell=True)
  subprocess.call(f"rm -f {simfilepol}", shell=True)
  # OLD VERSION (for debugging) subprocess.call(f"revan -g {params[2]} -c {params[8]} -f {simfileunpol} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  autorun(f"revan -g {params[2]} -c {params[8]} -f {simfileunpol} -n -a", f"{simname.split('/sim/')[0]}/unpol_revan_errlog.txt", trafileunpol)
  # Moving the cosima unpol file in rawsim or removing it
  # subprocess.call(f"mv {simfileunpol} {mv_simfileunpol}", shell=True)
  subprocess.call(f"rm -f {simfileunpol}", shell=True)

  # Running mimrec
  # OLD VERSION (for debugging) subprocess.call(f"mimrec -g {params[2]} -c {params[9]} -f {trafilepol} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  autorun(f"mimrec -g {params[2]} -c {params[9]} -f {trafilepol} -x -n", f"{simname.split('/sim/')[0]}/pol_mimrec_errlog.txt", extrfilepol)
  # Moving the revan analyzed pol file in rawsim or removing it
  # subprocess.call(f"mv {trafilepol} {mv_trafilepol}", shell=True)
  subprocess.call(f"rm -f {trafilepol}", shell=True)
  # OLD VERSION (for debugging) subprocess.call(f"mimrec -g {params[2]} -c {params[9]} -f {trafileunpol} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  autorun(f"mimrec -g {params[2]} -c {params[9]} -f {trafileunpol} -x -n", f"{simname.split('/sim/')[0]}/unpol_mimrec_errlog.txt", extrfileunpol)
  # Moving the revan analyzed unpol file in rawsim or removing it
  # subprocess.call(f"mv {trafileunpol} {mv_trafileunpol}", shell=True)
  subprocess.call(f"rm -f {trafileunpol}", shell=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Multi-threaded automated MEGAlib runner. Parse a parameter file (mono-threaded) to generate commands that are executed by cosima and revan in a multi-threaded way.")
  parser.add_argument("-f", "--parameterfile", help="Path to parameter file used to generate commands")
  args = parser.parse_args()
  if args.parameterfile:
    # Reading the param file
    print(f"Running of {args.parameterfile} parameter file")
    geometry, revanfile, mimrecfile, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = read_mupar(args.parameterfile)

    # Creating the required directories
    make_directories(geometry)
    # Calculating the flux corresponding to the spectrum, ampl is taken so that flux = 10cm²/s
    band_flux = use_scipyquad(band, 10, 1000, func_args=(bandparam[0], bandparam[1], bandparam[2], bandparam[3], bandparam[4]), x_logscale=True)[0]
    # Creating the parameter list
    parameters = make_parameters(decs, ras, geometry, source_base, spectra, poltime, unpoltime, band_flux, revanfile, mimrecfile)
    print("===================================================================")
    print(f"{len(parameters)} Commands have been parsed")
    print("===================================================================")

    print("===================================================================")
    print("Running the creation of GRB spectrum")
    print("===================================================================")
    make_spectrum(spectra, bandparam)
    print("===================================================================")
    print("Running the mu100 simulations and extraction")
    print("===================================================================")
    with mp.Pool() as pool:
      pool.map(run_mu, parameters)
  else:
    print("Missing parameter file or geometry - not running.")
