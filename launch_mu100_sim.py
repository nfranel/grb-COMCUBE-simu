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
from funcmod import band

def read_par(parfile):
  with open(parfile, "r") as f:
    lines = f.read().split("\n")
  geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = None, None, None, None, None, None, None, None, None, None
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
      if spectra.endswith("/"):
        spectra = spectra[:-1]
    elif line.startswith("@bandparam"):
      bandparam = list(map(float, line.split(" ")[1:]))
    elif line.startswith("@poltime"):
      poltime = float(line.split(" ")[1])
    elif line.startswith("@unpoltime"):
      unpoltime = float(line.split(" ")[1])
    elif line.startswith("@decposition"):
      decs = list(map(int, line.split(" ")[1:]))
      decs = np.linspace(decs[0], decs[1], decs[2])
    elif line.startswith("@raposition"):
      ras = list(map(int, line.split(" ")[1:]))
      ras = np.linspace(ras[0], ras[1], ras[2])
  return geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras


def make_directories(geometry):
  # Creating a directory specific to the geometry
  geom_name = geometry.split(".geo.setup")[0].split("/")[-1]
  if not f"sim_{geom_name}" in os.listdir("./mu100"):
    os.mkdir(f"./mu100/sim_{geom_name}")
    # Creating the sim and rawsim repertories if they don't exist
  if not f"sim" in os.listdir(f"./mu100/sim_{geom_name}"):
    os.mkdir(f"./mu100/sim_{geom_name}/sim")
  if not f"rawsim" in os.listdir(f"./mu100/sim_{geom_name}"):
    os.mkdir(f"./mu100/sim_{geom_name}/rawsim")


def make_spectrum(filepath, bandpar):
  """

  """
  if not (f"{filepath}/Band_spectrum.dat" in os.listdir(filepath)):
    logE = np.logspace(1, 3, 100)  # energy (log scale)
    with open(f"{filepath}/Band_spectrum.dat", "w") as f:
      f.write("#model band:  ")
      f.write(f"ampl={bandpar[0]}ph/cm2/keV/s, alpha={bandpar[1]}, beta={bandpar[2]}, epeak={bandpar[3]}keV, epivot={bandpar[4]}keV\n")
      f.write("\nIP LOGLOG\n\n")
      for E in logE:
        f.write(f"DP {E} {band(E, bandpar[0], bandpar[1], bandpar[2], bandpar[3], bandpar[4])}\n")
      f.write("\nEN\n\n")


def make_tmp_source(dec, ra, geom, source_model, spectra, timepol, timeunpol, ampl):
  """

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
        f.write(f"{source}.Spectrum File {spectra}/Band_spectrum.dat")
      elif line.startswith(f"{source}.Flux") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Flux {ampl}")
      else:
        f.write(line)
      f.write("\n")
  return fname, sname


def make_parameters(decs, ras, geom, source_model, spectra, timepol, timeunpol, ampl, rcffile, mimfile):
  """

  """
  parameters = []
  for dec in decs:
    for ra in ras:
      parameters.append((dec, ra, geom, source_model, spectra, timepol, timeunpol, ampl, rcffile, mimfile))
  return parameters


def run_bkg(params):
  # Making a temporary source file using a source_model
  sourcefile, simname = make_tmp_source(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
  # Making a generic name for files
  simfilepol, trafilepol = f"{simname}pol.inc1.id1.sim.gz", f"{simname}pol.inc1.id1.tra.gz"
  simfileunpol, trafileunpol = f"{simname}unpol.inc1.id1.sim.gz", f"{simname}unpol.inc1.id1.tra.gz"
  mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  mv_simfilepol, mv_trafilepol = f"{mv_simname}pol.inc1.id1.sim.gz", f"{mv_simname}pol.inc1.id1.tra.gz"
  mv_simfileunpol, mv_trafileunpol = f"{mv_simname}unpol.inc1.id1.sim.gz", f"{mv_simname}unpol.inc1.id1.tra.gz"
  #   Running the different simulations
  # Running cosima
  # subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  # subprocess.call(f"cosima -z {sourcefile}; rm -f {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'))
  subprocess.call(f"cosima -z {sourcefile}", shell=True, stdout=open(os.devnull, 'wb'))
  # Running revan
  # subprocess.call(f"revan -g {params[2]} -c {params[3]} -f {simfile} -n -a; rm -f {simfile}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  subprocess.call(f"revan -g {params[2]} -c {params[8]} -f {simfilepol} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  subprocess.call(f"mv {simfilepol} {mv_simfilepol}", shell=True)
  subprocess.call(f"revan -g {params[2]} -c {params[8]} -f {simfileunpol} -n -a", shell=True, stdout=open(os.devnull, 'wb'))
  subprocess.call(f"mv {simfileunpol} {mv_simfileunpol}", shell=True)
  # Running mimrec
  subprocess.call(f"mimrec -g {params[2]} -c {params[9]} -f {trafilepol} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  subprocess.call(f"mv {trafilepol} {mv_trafilepol}", shell=True)
  subprocess.call(f"mimrec -g {params[2]} -c {params[9]} -f {trafileunpol} -x -n", shell=True, stdout=open(os.devnull, 'wb'))
  subprocess.call(f"mv {trafileunpol} {mv_trafileunpol}", shell=True)


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
    geometry, revanfile, mimrecfile, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = read_par(args.parameterfile)

    # Creating the required directories
    make_directories(geometry)
    # Creating the parameter list
    parameters = make_parameters(decs, ras, geometry, source_base, spectra, poltime, unpoltime, bandparam[0], revanfile, mimrecfile)
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
    print("Running the background simulations and extraction")
    print("===================================================================")
    with mp.Pool() as pool:
      pool.map(run_bkg, parameters)
  else:
    print("Missing parameter file or geometry - not running.")

