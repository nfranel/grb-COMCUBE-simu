# Autor Nathan Franel
# Date 06/12/2023
# Version 2 :
# file to launch mu100 simulations

# Package imports
import subprocess
import os
import numpy as np
import multiprocessing as mp
import argparse
# Developped modules imports
from funcmod import band, read_mupar, quad


def make_directories(geomfile):
  """
  Create the directories in which the simulations are saved
  :param geomfile: geometry used for the simulations
  """
  # Creating a directory specific to the geometry
  geom_name = geomfile.split(".geo.setup")[0].split("/")[-1]
  if f"sim_{geom_name}" not in os.listdir("./mu100"):
    os.mkdir(f"./mu100/sim_{geom_name}")
    # Creating the sim and rawsim repertories if they don't exist
  if f"sim" not in os.listdir(f"./mu100/sim_{geom_name}"):
    os.mkdir(f"./mu100/sim_{geom_name}/sim")
  if f"rawsim" not in os.listdir(f"./mu100/sim_{geom_name}"):
    os.mkdir(f"./mu100/sim_{geom_name}/rawsim")


def make_spectrum(filepath, bandpar):
  """
  Create a band spectrum representing an average grb
  :param filepath: name of the file where the spectrum is saved
  :param bandpar: parameters for band spectrum
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
  Creates a temporary source file based on a model "source model"
  :param dec: dec for the mu100 simulation
  :param ra: ra for the mu100 simulation
  :param geom: geometry used for the mu100 simulation
  :param source_model: model used to create temporary source files
  :param spectrapath: path to spectra folder
  :param timepol: duration of the mu100 polarized simulation
  :param timeunpol: duration of the mu100 unpolarized simulation
  :param flux: flux for the simulations
  :returns: name of the temporary source file, name of the simulation without the extension
  """
  fname = f"tmp_{os.getpid()}.source"
  geom_name = geometry.split(".geo.setup")[0].split("/")[-1]
  sname = f"./mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}"
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
  Creates a list of right ascension for a specific dec : equator has more items than the poles (that have only 1)
  :param ra_list: list containing minimum ra, maximum ra and number of ra at equator [deg]
  :param dec: dec [deg]
  """
  if dec == 0 or dec == 180:
    new_ra = [0.0]
  else:
    new_ra = np.around(np.linspace(ra_list[0], ra_list[1], np.max([4, int(np.sin(np.deg2rad(dec)) * ra_list[2])]), endpoint=False), 1)
  return new_ra


def make_parameters(dec_list, ra_list, geomfile, source_model, spectrapath, timepol, timeunpol, flux, rcffile, mimfile):
  """
  Creates a lists of parameters for several altitudes and latitudes
  :param dec_list: decs for the mu100 simulation
  :param ra_list: ras for the mu100 simulation
  :param geomfile: geometry used for the mu100 simulation
  :param source_model: model used to create temporary source files
  :param spectrapath: path to spectra folder
  :param timepol: duration of the mu100 polarized simulation
  :param timeunpol: duration of the mu100 unpolarized simulation
  :param flux: flux for the simulations
  :param rcffile: revan configuration file to treat raw simulations
  :param mimfile: mimrec configuration file to extract simulations treated with revan
  """
  parameters_container = []
  for dec in np.linspace(dec_list[0], dec_list[1], dec_list[2]):
    for ra in make_ra_list(ra_list, dec):
      parameters_container.append((dec, ra, geomfile, source_model, spectrapath, timepol, timeunpol, flux, rcffile, mimfile))
  return parameters_container


def run_mu(params):
  """
  Runs the cosima, revan and mimrec programs and either move to rawsim or remove the .sim.gz and .tra.gz files
  :param params: list of parameters to run the simulation
  """
  # Making a temporary source file using a source_model
  sourcefile, simname = make_tmp_source(params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                                        params[7])
  stop
  # Making a generic name for files
  simfilepol, trafilepol = f"{simname}pol.inc1.id1.sim.gz", f"{simname}pol.inc1.id1.tra.gz"
  simfileunpol, trafileunpol = f"{simname}unpol.inc1.id1.sim.gz", f"{simname}unpol.inc1.id1.tra.gz"
  mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  mv_simfilepol, mv_trafilepol = f"{mv_simname}pol.inc1.id1.sim.gz", f"{mv_simname}pol.inc1.id1.tra.gz"
  mv_simfileunpol, mv_trafileunpol = f"{mv_simname}unpol.inc1.id1.sim.gz", f"{mv_simname}unpol.inc1.id1.tra.gz"

  #   Running the different simulations
  print(f"Running mu100 simulation : {simname}")
  # Running cosima
  # subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'))
  # subprocess.call(f"cosima -z {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'))

  # Running revan
  # subprocess.call(f"revan -g {params[2]} -c {params[3]} -f {simfile} -n -a; rm -f {simfile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  subprocess.call(f"revan -g {params[2]} -c {params[8]} -f {simfilepol} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  # Moving the cosima pol file in rawsim or removing it
  # subprocess.call(f"mv {simfilepol} {mv_simfilepol}", shell=True)
  subprocess.call(f"rm -f {simfilepol}", shell=True)
  subprocess.call(f"revan -g {params[2]} -c {params[8]} -f {simfileunpol} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  # Moving the cosima unpol file in rawsim or removing it
  # subprocess.call(f"mv {simfileunpol} {mv_simfileunpol}", shell=True)
  subprocess.call(f"rm -f {simfileunpol}", shell=True)

  # Running mimrec
  subprocess.call(f"mimrec -g {params[2]} -c {params[9]} -f {trafilepol} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  # Moving the revan analyzed pol file in rawsim or removing it
  # subprocess.call(f"mv {trafilepol} {mv_trafilepol}", shell=True)
  subprocess.call(f"rm -f {trafilepol}", shell=True)
  subprocess.call(f"mimrec -g {params[2]} -c {params[9]} -f {trafileunpol} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  # Moving the revan analyzed unpol file in rawsim or removing it
  # subprocess.call(f"mv {trafileunpol} {mv_trafileunpol}", shell=True)
  subprocess.call(f"rm -f {trafileunpol}", shell=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Multi-threaded automated MEGAlib runner. Parse a parameter file (mono-threaded) to generate commands that are executed by cosima and revan in a multi-threaded way.")
  parser.add_argument("-f", "--parameterfile", help="Path to parameter file used to generate commands")
  # parser.add_argument("-nc", "--nocosima", help="Does not run cosima", action="store_true")
  # parser.add_argument("-nr", "--norevan", help="Does not run revan", action="store_true")
  # parser.add_argument("-nm", "--nomimrec", help="Does not run mimrec", action="store_true")
  args = parser.parse_args()
  if args.parameterfile:
    # Reading the param file
    print(f"Running of {args.parameterfile} parameter file")
    geometry, revanfile, mimrecfile, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = read_mupar(
      args.parameterfile)

    # Creating the required directories
    make_directories(geometry)
    # Calculating the flux corresponding to the spectrum, ampl is taken so that flux = 10cm²/s
    func = lambda x: band(x, bandparam[0], bandparam[1], bandparam[2], bandparam[3], bandparam[4])
    band_flux = quad(func, 10, 1000)[0]
    # Creating the parameter list
    parameters = make_parameters(decs, ras, geometry, source_base, spectra, poltime, unpoltime, band_flux, revanfile,
                                 mimrecfile)
    print("===================================================================")
    print(f"{len(parameters)} Commands have been parsed")
    print("===================================================================")

    # Making the different sources spectra :
    # with mp.Pool() as pool:
    #   pool.map(make_spectra, parameters)
    # for params in parameters:
    #   make_spectra(params[6], params[0], params[1])
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
