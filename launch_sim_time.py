"""
Multi-threaded automated MEGAlib runner
"""

__version__ = "alpha"
__author__ = "Nathan Franel"

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import subprocess

import os
import argparse


__verbose__ = 1


### Utility functions

def vprint(s, v, l):
  """
  Prints string s if verbosity v is superior to verbosity level l
  """
  if v>l: print(s)


### Physics functions and class Catalog

from funcmod import *
from catalog import Catalog

### System functions

def genCommands(args):
  """
  Parses parameter file and fills Namespace with the data gathered
  """
  args.commands = []
  with open(args.parameterfile) as f:
    lines = f.read().split("\n")
  for line in lines:
    #general information
    if line.startswith("@type"): args.type = line.split(" ")[1]
    elif line.startswith("@cosimasourcefile"): args.csf = line.split(" ")[1]
    elif line.startswith("@revancfgfile"): args.rcf = line.split(" ")[1]
    elif line.startswith("@geometry") and args.geometry is None: args.geometry = line.split(" ")[1]
    elif line.startswith("@prefix") and args.prefix is None: args.prefix = line.split(" ")[1]
    #grb catalog
    elif line.startswith("@instrument"): args.instrument = line.split(" ")[1]
    elif line.startswith("@mode"): args.mode = line.split(" ")[1]
    elif line.startswith("@sttype"): args.sttype = line.split(" ")[1:]
    elif line.startswith("@file"): args.file = line.split(" ")[1]
    elif line.startswith("@simulationsperevent"): args.simulationsperevent = int(line.split(" ")[1])
    elif line.startswith("@poltime"): args.poltime = line.split(" ")[1]
    elif line.startswith("@unpoltime"): args.unpoltime = int(line.split(" ")[1])
    elif line.startswith("@position"): args.position = [float(e) for e in line.split(" ")[1:]] #sky positions (rad)
    elif line.startswith("@spectrafilepath"): args.spectrafilepath = line.split(" ")[1]
    elif line.startswith("@satellite"):
      temp = [float(e) for e in line.split(" ")[1:]]
      if len(temp) == 4:#satellite orbital parameters
        # Extracting inclination, ohm, omega, respectively the inclination, the right ascention of the ascending node and the true anomalie
        inclination, ohm, omega, alt = temp
        dat = [inclination, ohm, omega, alt]
      else:
        print("Wrong number of orbital parameters, please check that 4 values are given")
        raise ValueError
      if hasattr(args, "satellites"): args.satellites.append(dat)
      else: args.satellites = [dat]
  if args.type == "GRBCatalog":
    missing, attributes = [], ["instrument", "sttype", "file", "spectrafilepath", "simulationsperevent", "position", "satellite"]
    for a in attributes:
      if not(hasattr(args, a)):
        missing.append(a)
    if args.geometry is None: missing.append("geometry")
    if args.prefix is None: missing.append("prefix")
    if len(missing) != 1:
      raise AttributeError("One or several key-words are missing for a GRB catalog simulation - see list below. Check your parameter file.\n\n{}".format(missing))
    if args.instrument == "GBM":
      c = Catalog(args.file, args.sttype)
      vprint("Running with GBM data on {} mode.".format(args.mode), __verbose__, 0)
      items = "{0}_plaw_ampl,{0}_plaw_pivot,{0}_plaw_index,{0}_plaw_phtflux,{0}_comp_ampl,{0}_comp_epeak,{0}_comp_index,{0}_comp_pivot,{0}_comp_phtflux,{0}_band_ampl,{0}_band_epeak,{0}_band_alpha,{0}_band_beta,{0}_band_phtflux,{0}_sbpl_ampl,{0}_sbpl_pivot,{0}_sbpl_indx1,{0}_sbpl_brken,{0}_sbpl_brksc,{0}_sbpl_indx2,{0}_sbpl_phtflux".format(args.mode).split(",")
      defaults = [0, 100, 0, 0, 0, 1, 0, 100, 0, 0, 1, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0]
      c.tofloats(items, defaults)
      args.commands = []
      with open("simulation_logs.txt", "w") as f:
        f.write("========================================================================")
        f.write("                    Log file for the simulations                        ")
        f.write("GRB name | simulation number | satellite number | status of the simulation | sat inclination | sat RA of ascending node | sat argument of periapsis | altitude | sat dec world frame | sat ra world frame | grb dec world frame | grb ra world frame | grb dec sat frame | grb ra sat frame")
        f.write("Angles in degrees and altitude in km")
        f.write("========================================================================")
      def genGRB(i): #Generate a single GRB command
        #generate spectrum file here if not already done
        if not(args.spectrafilepath.endswith("/")) and os.name == "posix": args.spectrafilepath += "/"
        spectrumfile = "{}{}_spectrum.dat".format(args.spectrafilepath, c.name[i])
        model = getattr(c, "{}_best_fitting_model".format(args.mode))[i].strip()
        phtflux = getattr(c, "{}_phtflux".format(model))[i]
        pht_mflx = getattr(c, "{}_phtflux".format(getattr(c, "flnc_best_fitting_model")[i].strip()))[i]
        pht_pflx = getattr(c, "{}_phtflux".format(getattr(c, "pflx_best_fitting_model")[i].strip()))[i]
        if not(spectrumfile in os.listdir(args.spectrafilepath)):
          logE = np.logspace(1, 3, 100)#energy (log scale)
          with open(spectrumfile, "w") as f:
            f.write("#model {}:  ".format(model))
            if model == "pflx_plaw":
              fun = lambda x: plaw(x, c.pflx_plaw_ampl[i], c.pflx_plaw_index[i], c.pflx_plaw_pivot[i])
              f.write("ampl={}, index={}, pivot={}keV\n".format(c.pflx_plaw_ampl[i], c.pflx_plaw_index[i], c.pflx_plaw_pivot[i]))
            elif model == "pflx_comp":
              fun = lambda x: comp(x, c.pflx_comp_ampl[i], c.pflx_comp_index[i], c.pflx_comp_epeak[i], c.pflx_comp_pivot[i])
              f.write("ampl={}, index={}, epeak={}keV, pivot={}keV\n".format(c.pflx_comp_ampl[i], c.pflx_comp_index[i], c.pflx_comp_epeak[i], c.pflx_comp_pivot[i]))
            elif model == "pflx_band":
              fun = lambda x: band(x, c.pflx_band_ampl[i], c.pflx_band_alpha[i], c.pflx_band_beta[i], c.pflx_band_epeak[i])
              f.write("ampl={}, alpha={}, beta={}, epeak={}keV\n".format(c.pflx_band_ampl[i], c.pflx_band_alpha[i], c.pflx_band_beta[i], c.pflx_band_epeak[i]))
            elif model == "pflx_sbpl":
              fun = lambda x: sbpl(x, c.pflx_sbpl_ampl[i], c.pflx_sbpl_indx1[i], c.pflx_sbpl_indx2[i], c.pflx_sbpl_brken[i], c.pflx_sbpl_brksc[i], c.pflx_sbpl_pivot[i])
              f.write("ampl={}, index1={}, index2={}, eb={}keV, brksc={}keV, pivot={}keV\n".format(c.pflx_sbpl_ampl[i], c.pflx_sbpl_indx1[i], c.pflx_sbpl_indx2[i], c.pflx_sbpl_brken[i], c.pflx_sbpl_brksc[i], c.pflx_sbpl_pivot[i]))
            elif model == "flnc_plaw":
              fun = lambda x: plaw(x, c.flnc_plaw_ampl[i], c.flnc_plaw_index[i], c.flnc_plaw_pivot[i])
              f.write("ampl={}, index={}, pivot={}keV\n".format(c.flnc_plaw_ampl[i], c.flnc_plaw_index[i], c.flnc_plaw_pivot[i]))
            elif model == "flnc_comp":
              fun = lambda x: comp(x, c.flnc_comp_ampl[i], c.flnc_comp_index[i], c.flnc_comp_epeak[i], c.flnc_comp_pivot[i])
              f.write("ampl={}, index={}, epeak={}keV, pivot={}keV\n".format(c.flnc_comp_ampl[i], c.flnc_comp_index[i], c.flnc_comp_epeak[i], c.flnc_comp_pivot[i]))
            elif model == "flnc_band":
              fun = lambda x: band(x, c.flnc_band_ampl[i], c.flnc_band_alpha[i], c.flnc_band_beta[i], c.flnc_band_epeak[i])
              f.write("ampl={}, alpha={}, beta={}, epeak={}keV\n".format(c.flnc_band_ampl[i], c.flnc_band_alpha[i], c.flnc_band_beta[i], c.flnc_band_epeak[i]))
            elif model == "flnc_sbpl":
              fun = lambda x: sbpl(x, c.flnc_sbpl_ampl[i], c.flnc_sbpl_indx1[i], c.flnc_sbpl_indx2[i], c.flnc_sbpl_brken[i], c.flnc_sbpl_brksc[i], c.flnc_sbpl_pivot[i])
              f.write("ampl={}, index1={}, index2={}, eb={}keV, brksc={}keV, pivot={}keV\n".format(c.flnc_sbpl_ampl[i], c.flnc_sbpl_indx1[i], c.flnc_sbpl_indx2[i], c.flnc_sbpl_brken[i], c.flnc_sbpl_brksc[i], c.flnc_sbpl_pivot[i]))
            else:
              vprint("Could not find best fit model for {} (indicated {}). Aborting this GRB.".format(c.name[i], model), __verbose__, 0)
              return
            f.write("# Measured mean flux: {} ph/cm2/s in the 10-1000 keV band\n".format(pht_mflx))
            f.write("# Measured peak flux: {} ph/cm2/s in the 10-1000 keV band\n".format(pht_pflx))
            f.write("\nIP LOGLOG\n\n")
            for E in logE:
              f.write("DP {} {}\n".format(E, fun(E)))
            f.write("\nEN\n\n")
        for j in range(args.simulationsperevent):
          dec_grb_world_frame, ra_grb_world_frame = random_GRB_dec_ra(args.position[0], args.position[1], args.position[2], args.position[3]) # deg
          rand_time = np.random.rand()*315567360.0 # Time of the GRB, taken randomly over a 10 years time window
          for k, s in enumerate(args.satellites):
            orbital_period = orbital_period_calc(s[3])
            earth_ra_offset = earth_rotation_offset(rand_time)
            true_anomaly = true_anomaly_calc(rand_time, orbital_period)
            dec_sat_world_frame, ra_sat_world_frame = orbitalparam2decra(s[0], s[1], s[2], nu=true_anomaly) #deg
            ra_sat_world_frame -= earth_ra_offset
            print(orbital_period, rand_time, earth_ra_offset, true_anomaly)
            print(dec_sat_world_frame, ra_sat_world_frame)
            if verif_zone(90 - dec_sat_world_frame, ra_sat_world_frame):  # checks if the satellite is in the switch off zone
              save_log("simulation_logs.txt", c.name[i], j, k, "Ignored(off)", s[0], s[1], s[2], s[3], dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, None, None)
            else:
              theta, phi, thetap, phip = grb_decrapol_worldf2satf(dec_grb_world_frame, ra_grb_world_frame, dec_sat_world_frame, ra_sat_world_frame)[1:]
              if theta >= horizonAngle(temp[3]):#source below horizon
                save_log("simulation_logs.txt", c.name[i], j, k, "Ignored(horizon)", s[0], s[1], s[2], s[3], dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
              else:
                polstr = "{} {} {}".format(np.sin(thetap)*np.cos(phip), np.sin(thetap)*np.sin(phip), np.cos(thetap))
                # Add command to commands list
                if args.poltime.isdigit():
                  poltime = float(args.poltime)
                elif args.poltime == "t90":
                  poltime = float(c.t90[i])
                else:
                  vprint("Poltime in parameter file unknown. Check parameter file.", __verbose__, 0)
                unpoltime = args.unpoltime
                args.commands.append((not(args.nocosima), not(args.norevan), c.name[i], k, spectrumfile, phtflux, poltime, unpoltime, polstr, j, "{:.1f}_{:.1f}".format(np.rad2deg(dec_grb_world_frame), np.rad2deg(ra_grb_world_frame)), theta, phi))
                save_log("simulation_logs.txt", c.name[i], j, k, "Simulated", s[0], s[1], s[2], s[3], dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
      for i in range(len(c)): genGRB(i)
  else: vprint("Type in parameter file unknown. Check parameter file.", __verbose__, 0)
  return args


def makeSimName(args, command):
  """
  Makes the beginning of the .sim file name from args and command
  """
  if args.type == "EffectiveArea":
    if hasattr(args, "angle"):  return "{}_{}_{}".format(args.prefix, command[2], command[3])
    else: return "{}_{}_{}_{}".format(args.prefix, command[2], command[3], command[4])
  elif args.type == "GRBCatalog":
    #return "{}_{}-sat{}_{}_{}".format(args.prefix, command[2], command[3], command[-2], command[-1])
    return "{}_{}_sat{}_{:04d}_{}".format(args.prefix, command[2], command[3], command[-4], command[-3])


def maketmpsf(command, args, pid):
  """
  Makes a temporary source file for cosima from a standard one and returns its name
  """
  fname = "tmp_{}.source".format(pid)
  sname = makeSimName(args, command)
  if args.type == "GRBCatalog":
    c = Catalog(args.file, args.sttype)
    with open(args.csf) as f:
      lines = f.read().split("\n")
    with open(fname, "w") as f:
      run, source = "", ""#"GRBsource" or "GRBsourcenp"
      for line in lines:
        if line.startswith("Geometry"):
          f.write("Geometry {}".format(args.geometry))
        elif line.startswith("PhysicsListEM "):
          f.write(line)
          if line.split(" ")[-1] != "Livermore-Pol":
            vprint("Warning : the PhysicsListEM used does not handle polarization, results may be impacted", __verbose__, 0)
        elif line.startswith("Run"):
          run = line.split(" ")[-1]
          if run == "GRBpol" or run == "GRBnpol":
            f.write(line)
          else:
            vprint("Name of run is not valid. Check parameter file and use either GRBpol for polarized run or GRBnpol for unpolarized run.", __verbose__, 0)
        elif line.startswith("{}.Source".format(run)):
          source = line.split(" ")[-1]
          if source == "GRBsource" or source == "GRBsourcenp":
            f.write(line)
          else:
            vprint("Name of source is not valid. Check parameter file and use either GRBsource for polarized run or GRBsourcenp for unpolarized run.", __verbose__, 0)
        # Adding 2 more conditions to prevent background sources to be changed
        elif line.startswith("{}.FileName".format(run)):
          f.write("{}.FileName {}".format(run, sname))
        elif line.startswith("{}.Time".format(run)):
          if run == "GRBpol":
            f.write("{}.Time {}".format(run, command[6]))
          elif run == "GRBnpol":
            f.write("{}.Time {}".format(run, command[7]))
        elif line.startswith("{}.Beam".format(source)) and (source=="GRBsource" or source=="GRBsourcenp"):
          f.write("{}.Beam FarFieldPointSource {} {}".format(source, command[-2], command[-1]))
        elif line.startswith("{}.Spectrum".format(source)) and (source=="GRBsource" or source=="GRBsourcenp"):
          f.write("{}.Spectrum File {}".format(source, command[4]))
        elif line.startswith("{}.Polarization".format(source)) and (source=="GRBsource" or source=="GRBsourcenp"):
          f.write("{}.Polarization Absolute 1. {}".format(source, command[8]))
        elif line.startswith("{}.Flux".format(source)) and (source=="GRBsource" or source=="GRBsourcenp"):
          f.write("{}.Flux {}".format(source, command[5]))
        else: f.write(line)
        f.write("\n")
    return fname


### MEGAlib interface functions

def cosirevan(command):
  """
  Launches cosima and/or revan
  command syntax : tuple : command[0] - bool, run cosima : command[1] - run revan
  if both, runs revan only on the latest .sim file with the correct name
  """
  pid = os.getpid()
  if command[0]: #run cosima
    run("cosima -z {0}; rm -f {0}".format(maketmpsf(command, args, pid)), __verbose__)
  if command[1]: #run revan
    sname = makeSimName(args, command)
    if command[0]:
#!!!# Use the following two lines for polarization sensitivity evaluation
#!!!# 2 lines because 2 runs if polarization, so a first run is read and suppressed and then then second one
      run("revan -g {0} -c {1} -f {2} -n -a; rm -f {2}".format(args.geometry, args.rcf, subprocess.getoutput("ls -t {}*.sim.gz".format(sname)).split("\n")[0]), __verbose__)
      run("revan -g {0} -c {1} -f {2} -n -a; rm -f {2}".format(args.geometry, args.rcf, subprocess.getoutput("ls -t {}*.sim.gz".format(sname)).split("\n")[0]), __verbose__)
#!!!# Use the following line for GRB detection rate evaluation
      #run("revan -g {0} -c {1} -f {2} -n -a".format(args.geometry, args.rcf, subprocess.getoutput("ls -t {}*.sim.gz".format(sname)).split("\n")[0]), __verbose__)
    else:
      files = subprocess.getoutput("ls {}*.sim.gz".format("_".join(sname.split("_")[:-2]))).split("\n")
      for f in files:
        run("revan -g {} -c {} -f {} -n -a".format(args.geometry, args.rcf, f), __verbose__)


def run(command, __verbose__):
  """
  Runs a command
  :param command: str, shell command to run
  :param __verbose__: verbosity
    0 -> No output
    1 -> One-line output (proscesses id, command and verbosity)
    2 -> Adds stderr of command
    3 -> Adds stdout of command
  """
  vprint("Process id {} from {} runs {} (verbosity {})".format(os.getpid(), os.getppid(), command, __verbose__), __verbose__, 0)
  if __verbose__ < 2: subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  elif __verbose__ < 3: subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb'))
  else: subprocess.call(command, shell=True)


def runSims(commands):
  """
  Run each command in commands through multiprocessing module
  """
  with mp.Pool() as pool:
    pool.map(cosirevan, commands)


### MAIN

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Multi-threaded automated MEGAlib runner. Parse a parameter file (mono-threaded) to generate commands that are executed by cosima and revan in a multi-threaded way.")
  parser.add_argument("-f", "--parameterfile", help="Path to parameter file used to generate commands")
  parser.add_argument("-g", "--geometry", help="Path to geometry file", default=None)
  parser.add_argument("-p", "--prefix", help="Output file prefix", default=None)
  parser.add_argument("-nc", "--nocosima", help="Does not run cosima", action="store_true")
  parser.add_argument("-nr", "--norevan", help="Does not run revan", action="store_true")
  parser.add_argument("-v", "--verbose", help="Verbosity level (0 to 3)", type=int, default=__verbose__)
  parser.add_argument("-V", "--version", help="Prints out the version of the script", action="store_true")
  args = parser.parse_args()
  if args.version:
    print("Script version {} written by {}.".format(__version__, __author__))
  if args.parameterfile:
    __verbose__ = args.verbose
    vprint("Running of {} parameter file with output prefix {}".format(args.parameterfile, args.prefix), __verbose__, 0)
    args = genCommands(args)
    vprint("{} Commands have been parsed".format(len(args.commands)), __verbose__, 0)
    runSims(args.commands)
  else:
    vprint("Missing parameter file or geometry - not running.", __verbose__, 0)



