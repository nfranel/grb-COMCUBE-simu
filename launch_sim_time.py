"""
Multi-threaded automated MEGAlib runner
"""
__version__ = "alpha"
__author__ = "Nathan Franel"

# Date 01/12/2023
# Version 2 :
# file to launch background simulations

# Package imports
import multiprocessing as mp
import subprocess
import os
import argparse
import astropy.units
# Developped modules imports
from funcmod import *
from funcsample import norm_band_spec_calc
from catalog import Catalog, SampleCatalog
# Useful constants
keV_to_erg = 1 * astropy.units.keV
keV_to_erg = keV_to_erg.to_value("erg")
Gpc_to_cm = 1 * astropy.units.Gpc
Gpc_to_cm = Gpc_to_cm.to_value("cm")

__verbose__ = 1


# Utility functions
def vprint(message, verb, level):
  """
  Prints string message if verbosity verb is superior to verbosity level level
  """
  if verb > level:
    print(message)


# System functions
def gen_commands(args):
  """
  Parses parameter file and fills Namespace with the data gathered
  """
  args.commands = []
  args.geometry, args.rcf, args.mcf, args.simmode, args.spectrafilepath, args.grbfile, args.csf, args.prefix, args.sttype, args.simulationsperevent, args.simtime, args.position, args.satellites = read_grbpar(args.parameterfile)
  cat_rest_info = "./GBM/rest_frame_properties.txt"
  if args.simmode == "GBM":
    cat = Catalog(args.grbfile, args.sttype, cat_rest_info)
    vprint("Running with GBM data on flnc mode.", __verbose__, 0)
  elif args.simmode == "sampled":
    cat = SampleCatalog(args.grbfile, args.sttype)
  else:
    raise ValueError("Wrong simulation mode in .par file")
  args.commands = []
  sim_directory = args.prefix.split("/sim/")[0]
  with open(f"{sim_directory}/simulation_logs.txt", "w") as f:
    f.write("========================================================================\n")
    f.write("                    Log file for the simulations                        \n")
    f.write("GRB name | simulation number | satellite number | status of the simulation | sat inclination | sat RA of ascending node | sat argument of periapsis | altitude | random time of simulation | sat dec world frame | sat ra world frame | grb dec world frame | grb ra world frame | grb dec sat frame | grb ra sat frame\n")
    f.write("name | sim_num | sat_num | status | inc | ohm | omega | alt | rand_time | sat_decwf | sat_rawf | grb_decwf | grb_rawf | grb_decsf | grb_rasf\n")
    f.write("Angles in degrees and altitude in km\n")
    f.write("========================================================================\n")
  def gen_grb(i): # Generate a single GRB command
    # generate spectrum file here if not already done
    if not (args.spectrafilepath.endswith("/")) and os.name == "posix":
      args.spectrafilepath += "/"
    if args.simmode == "sampled":
      lc_name = cat.df.lc[i]
      pht_mflx = cat.df.mean_flux[i]
      n_year = float(args.grbfile.split("years")[0].split("_")[-1])
      spectrafolder = f"{args.spectrafilepath}{n_year}sample/"
      spectrumfile = "{}{}_spectrum.dat".format(spectrafolder, cat.df.name[i])
      if not (f"{n_year}sample" in os.listdir(args.spectrafilepath)):
        os.mkdir(spectrafolder)
      # Creation of spectra if they have not been created yet
      if not (spectrumfile in spectrafolder):
        logE = np.logspace(1, 3, 100)  # energy (log scale)
        with open(spectrumfile, "w") as f:
          norm_val, spec, pht_pflux = norm_band_spec_calc(cat.df.alpha[i], cat.df.beta[i], cat.df.z_obs[i], cat.df.dl[i], cat.df.ep_rest[i], cat.df.liso[i], logE)
          f.write(f"#model normalized Band:   norm={norm_val}, alpha={cat.df.alpha[i]}, beta={cat.df.beta[i]}, epeak={cat.df.ep_rest[i]}keV\n")
          f.write(f"# Measured mean flux: {pht_mflx} ph/cm2/s in the 10-1000 keV band\n")
          f.write(f"# Measured peak flux: {pht_pflux} ph/cm2/s in the 10-1000 keV band\n")
          f.write("\nIP LOGLOG\n\n")
          for ite_E, E in enumerate(logE):
            f.write(f"DP {E} {spec[ite_E]}\n")
          f.write("\nEN\n\n")
    else:
      spectrumfile = "{}{}_spectrum.dat".format(args.spectrafilepath, cat.df.name[i])
      lc_name = None
      model = cat.df.flnc_best_fitting_model[i]
      pht_mflx = cat.df[f"{model}_phtflux"][i]
      pfluxmodel = cat.df.pflx_best_fitting_model[i]
      if type(pfluxmodel) == str:
        pht_pflx = cat.df[f"{pfluxmodel}_phtflux"][i]
      else:
        if np.isnan(pfluxmodel):
          pht_pflx = "No value fitted"
        else:
          raise ValueError("A value for pflx_best_fitting_model is not set properly")
      # Creation of spectra if they have not been created yet
      if not (spectrumfile in os.listdir(args.spectrafilepath)):
        logE = np.logspace(1, 3, 100)  # energy (log scale)
        with open(spectrumfile, "w") as f:
          f.write("#model {}:  ".format(model))
          if model == "flnc_plaw":
            func_args = (cat.df.flnc_plaw_ampl[i], cat.df.flnc_plaw_index[i], cat.df.flnc_plaw_pivot[i])
            # fun = lambda x: plaw(x, cat.flnc_plaw_ampl[i], cat.flnc_plaw_index[i], cat.flnc_plaw_pivot[i])
            fun = plaw
            f.write(f"ampl={func_args[0]}, index={func_args[1]}, pivot={func_args[2]}keV\n")
          elif model == "flnc_comp":
            func_args = (cat.df.flnc_comp_ampl[i], cat.df.flnc_comp_index[i], cat.df.flnc_comp_epeak[i], cat.df.flnc_comp_pivot[i])
            # fun = lambda x: comp(x, cat.flnc_comp_ampl[i], cat.flnc_comp_index[i], cat.flnc_comp_epeak[i], cat.flnc_comp_pivot[i])
            fun = comp
            f.write(f"ampl={func_args[0]}, index={func_args[1]}, epeak={func_args[2]}keV, pivot={func_args[3]}keV\n")
          elif model == "flnc_band":
            func_args = (cat.df.flnc_band_ampl[i], cat.df.flnc_band_alpha[i], cat.df.flnc_band_beta[i], cat.df.flnc_band_epeak[i])
            # fun = lambda x: band(x, cat.flnc_band_ampl[i], cat.flnc_band_alpha[i], cat.flnc_band_beta[i], cat.flnc_band_epeak[i])
            fun = band
            f.write(f"ampl={func_args[0]}, alpha={func_args[1]}, beta={func_args[2]}, epeak={func_args[3]}keV\n")
          elif model == "flnc_sbpl":
            func_args = (cat.df.flnc_sbpl_ampl[i], cat.df.flnc_sbpl_indx1[i], cat.df.flnc_sbpl_indx2[i], cat.df.flnc_sbpl_brken[i], cat.df.flnc_sbpl_brksc[i], cat.df.flnc_sbpl_pivot[i])
            # fun = lambda x: sbpl(x, cat.flnc_sbpl_ampl[i], cat.flnc_sbpl_indx1[i], cat.flnc_sbpl_indx2[i], cat.flnc_sbpl_brken[i], cat.flnc_sbpl_brksc[i], cat.flnc_sbpl_pivot[i])
            fun = sbpl
            f.write(f"ampl={func_args[0]}, index1={func_args[1]}, index2={func_args[2]}, eb={func_args[3]}keV, brksc={func_args[4]}keV, pivot={func_args[5]}keV\n")
          else:
            vprint(f"Could not find best fit model for {cat.df.name[i]} (indicated {model}). Aborting this GRB.", __verbose__, 0)
            return
          f.write(f"# Measured mean flux: {pht_mflx} ph/cm2/s in the 10-1000 keV band\n")
          f.write(f"# Measured peak flux: {pht_pflx} ph/cm2/s in the 10-1000 keV band\n")
          f.write("\nIP LOGLOG\n\n")
          for E in logE:
            f.write(f"DP {E} {fun(E, *func_args)}\n")
          f.write("\nEN\n\n")
    for j in range(args.simulationsperevent):
      dec_grb_world_frame, ra_grb_world_frame = random_grb_dec_ra(args.position[0], args.position[1], args.position[2], args.position[3])  # deg
      rand_time = np.around(np.random.rand()*315567360.0, 4)  # Time of the GRB, taken randomly over a 10 years time window
      for k, s in enumerate(args.satellites):
        orbital_period = orbital_period_calc(s[3])
        earth_ra_offset = earth_rotation_offset(rand_time)
        true_anomaly = true_anomaly_calc(rand_time, orbital_period)
        dec_sat_world_frame, ra_sat_world_frame = orbitalparam2decra(s[0], s[1], s[2], nu=true_anomaly)  # deg
        ra_sat_world_frame = np.mod(ra_sat_world_frame - earth_ra_offset, 360)
        if verif_rad_belts(dec_sat_world_frame, ra_sat_world_frame, s[3]):  # checks if sat is in the switch off zone
          save_log(f"{sim_directory}/simulation_logs.txt", cat.df.name[i], j, k, "Ignored(off)", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, None, None)
        else:
          theta, phi, thetap, phip, polstr = grb_decrapol_worldf2satf(dec_grb_world_frame, ra_grb_world_frame, dec_sat_world_frame, ra_sat_world_frame)[1:]
          if theta >= horizon_angle(s[3]):  # Source below horizon
            save_log(f"{sim_directory}/simulation_logs.txt", cat.df.name[i], j, k, "Ignored(horizon)", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
          else:
            # Add command to commands list
            if args.simtime.isdigit():
              simtime = float(args.simtime)
              lc_bool = False
            elif args.simtime == "t90":
              simtime = cat.df.t90[i]
              lc_bool = False
            elif args.simtime == "lc":
              simtime = cat.df.t90[i]
              lc_bool = True
            else:
              simtime = None
              lc_bool = False
              vprint("simtime in parameter file unknown. Check parameter file.", __verbose__, 0)
            args.commands.append((not(args.nocosima), not(args.norevan), not(args.nomimrec), cat.df.name[i], k, spectrumfile, pht_mflx, simtime, lc_bool, lc_name, polstr, j, f"{dec_grb_world_frame:.4f}_{ra_grb_world_frame:.4f}_{rand_time:.4f}", theta, phi))
            save_log(f"{sim_directory}/simulation_logs.txt", cat.df.name[i], j, k, "Simulated", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
  for i in range(len(cat.df)):
    gen_grb(i)
  return args


def make_sim_name(args, command):
  """
  Makes the beginning of the .sim file name from args and command
  """
  return f"{args.prefix}_{command[3]}_sat{command[4]}_{command[-4]:04d}_{command[-3]}"


def maketmpsf(command, args, pid):
  """
  Makes a temporary source file for cosima from a standard one and returns its name
  """
  fname = f"tmp_{pid}.source"
  sname = make_sim_name(args, command)
  with open(args.csf) as f:
    lines = f.read().split("\n")
  with open(fname, "w") as f:
    run, source = "", ""  # So that the test with startswith may be done even if run and source are not initialized yet
    for line in lines:
      if line.startswith("Geometry"):
        f.write(f"Geometry {args.geometry}")
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
      elif line.startswith(f"{run}.Source"):
        source = line.split(" ")[-1]
        if source == "GRBsource" or source == "GRBsourcenp":
          f.write(line)
        else:
          vprint("Name of source is not valid. Check parameter file and use either GRBsource for polarized run or GRBsourcenp for unpolarized run.", __verbose__, 0)
      # Adding 2 more conditions to prevent background sources to be changed
      elif line.startswith(f"{run}.FileName"):
        f.write(f"{run}.FileName {sname}")
      elif line.startswith(f"{run}.Time"):
        f.write(f"{run}.Time {command[7]}")
      elif line.startswith(f"{source}.Beam") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Beam FarFieldPointSource {command[-2]} {command[-1]}")
      elif line.startswith(f"{source}.Spectrum") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Spectrum File {command[5]}")
      elif line.startswith(f"{source}.Polarization") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Polarization Absolute 1. {command[10]}")
      elif line.startswith(f"{source}.Flux") and (source == "GRBsource" or source == "GRBsourcenp"):
        f.write(f"{source}.Flux {command[6]}")
        if command[8]:
          if command[9] is None:
            f.write(f"\n{source}.Lightcurve File true ./sources/Light_Curves/LightCurve_{command[3]}.dat")
          else:
            f.write(f"\n{source}.Lightcurve File true ./sources/Light_Curves/{command[9]}")
      else:
        f.write(line)
      f.write("\n")
  return fname


# MEGAlib interface functions
def cosirevan(command):
  """
  Launches cosima and/or revan
  command syntax : tuple : command[0] - bool, run cosima : command[1] - run revan
  if both, runs revan only on the latest .sim file with the correct name
  """
  pid = os.getpid()
  simname = make_sim_name(args, command)
  simfile, trafile = f"{simname}.inc1.id1.sim.gz", f"{simname}.inc1.id1.tra.gz"
  mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  mv_simfile, mv_trafile = f"{mv_simname}.inc1.id1.sim.gz", f"{mv_simname}.inc1.id1.tra.gz"
  source_name = maketmpsf(command, args, pid)
  if command[0]:
    # Running cosima
    run(f"cosima -z {source_name}; rm -f {source_name}", __verbose__)
  if command[1]:
    # Running revan and moving the simulation file to rawsim
    # run(f"revan -g {args.geometry} -c {args.rcf} -f {simfile} -n -a; mv {simfile} {mv_simfile}", __verbose__)
    # Running revan and removing the simulation file
    run(f"revan -g {args.geometry} -c {args.rcf} -f {simfile} -n -a; rm -f {simfile}", __verbose__)
  if command[2]:
    # Running mimrec and moving the revan analyzed file to rawsim
    # run(f"mimrec -g {args.geometry} -c {args.mcf} -f {trafile} -x -n; mv {trafile} {mv_trafile}", __verbose__)
    # Running mimrec and removing the revan analyzed file
    run(f"mimrec -g {args.geometry} -c {args.mcf} -f {trafile} -x -n; rm -f {trafile}", __verbose__)


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
  if __verbose__ < 2:
    subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  elif __verbose__ < 3:
    subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb'))
  else:
    subprocess.call(command, shell=True)


def run_sims(commands):
  """
  Run each command in commands through multiprocessing module
  """
  with mp.Pool() as pool:
    pool.map(cosirevan, commands)


# MAIN
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Multi-threaded automated MEGAlib runner. Parse a parameter file (mono-threaded) to generate commands that are executed by cosima and revan in a multi-threaded way.")
  parser.add_argument("-f", "--parameterfile", help="Path to parameter file used to generate commands")
  parser.add_argument("-g", "--geometry", help="Path to geometry file", default=None)
  parser.add_argument("-p", "--prefix", help="Output file prefix", default=None)
  parser.add_argument("-nc", "--nocosima", help="Does not run cosima", action="store_true")
  parser.add_argument("-nr", "--norevan", help="Does not run revan", action="store_true")
  parser.add_argument("-nm", "--nomimrec", help="Does not run mimrec", action="store_true")
  parser.add_argument("-v", "--verbose", help="Verbosity level (0 to 3)", type=int, default=__verbose__)
  parser.add_argument("-V", "--version", help="Prints out the version of the script", action="store_true")
  args = parser.parse_args()
  if args.version:
    print("Script version {} written by {}.".format(__version__, __author__))
  if args.parameterfile:
    __verbose__ = args.verbose
    vprint("Running of {} parameter file with output prefix {}".format(args.parameterfile, args.prefix), __verbose__, 0)
    args = gen_commands(args)
    vprint("{} Commands have been parsed".format(len(args.commands)), __verbose__, 0)
    run_sims(args.commands)
  else:
    vprint("Missing parameter file or geometry - not running.", __verbose__, 0)



