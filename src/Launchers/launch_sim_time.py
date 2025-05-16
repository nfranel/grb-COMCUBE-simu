"""
Multi-threaded automated MEGAlib runner
"""
__version__ = "alpha"
__author__ = "Nathan Franel"

# Date 01/12/2023
# Version 2 :
# file to launch background simulations

# Package imports
import numpy as np
import multiprocessing as mp
import subprocess
import os
import argparse
import astropy.units

# Developped modules imports
from src.General.funcmod import norm_band_spec_calc, read_grbpar, make_sample_lc, plaw, comp, band, sbpl, random_grb_dec_ra, orbital_period_calc, earth_rotation_offset, true_anomaly_calc, orbitalparam2decra, verif_rad_belts, grb_decrapol_worldf2satf, horizon_angle, save_log
from src.Catalogs.catalog import Catalog, SampleCatalog
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
  cat_rest_info = "../Data/CatData/rest_frame_properties.txt"
  if args.simmode == "GBM":
    cat = Catalog(args.grbfile, args.sttype, cat_rest_info)
    vprint("Running with GBM data on flnc mode.", __verbose__, 0)
    make_lc_files = False
  elif args.simmode == "sampled":
    cat = SampleCatalog(args.grbfile, args.sttype)
    if not os.path.exists("../Data/sources/Sample_Light_Curves"):
      os.mkdir("../Data/sources/Sample_Light_Curves")
    make_lc_files = not (int(subprocess.getoutput("ls ../Data/sources/Sample_Light_Curves | wc").strip().split("  ")[0]) == len(cat.df))
  else:
    raise ValueError("Wrong simulation mode in .par file")
  args.commands = []
  sim_directory = args.prefix.split("/sim/")[0]
  with open(f"{sim_directory}/simulation_logs.txt", "w") as f:
    f.write("========================================================================\n")
    f.write("                    Log file for the simulations                        \n")
    f.write("GRB name | GRB index | simulation number | satellite number | status of the simulation | sat inclination | sat RA of ascending node | sat argument of periapsis | altitude | random time of simulation | sat dec world frame | sat ra world frame | grb dec world frame | grb ra world frame | grb dec sat frame | grb ra sat frame\n")
    f.write("name | grb_num | sim_num | sat_num | status | inc | ohm | omega | alt | rand_time | sat_decwf | sat_rawf | grb_decwf | grb_rawf | grb_decsf | grb_rasf\n")
    f.write("Angles in degrees and altitude in km\n")
    f.write("========================================================================\n")
  def gen_grb(i): # Generate a single GRB command
    # generate spectrum file here if not already done
    if not (args.spectrafilepath.endswith("/")) and os.name == "posix":
      args.spectrafilepath += "/"
    if args.simmode == "sampled":
      lc_name = cat.df.lc.values[i]
      pht_mflx = cat.df.mean_flux.values[i]
      pht_pflx = cat.df.peak_flux.values[i]
      n_year = float(args.grbfile.split("years")[0].split("_")[-1])
      spectrafolder = f"{args.spectrafilepath}{int(n_year)}sample/"
      spectrumfile = f"{spectrafolder}{cat.df.name.values[i]}_spectrum.dat"
      if not (f"{int(n_year)}sample" in os.listdir(args.spectrafilepath)):
        os.mkdir(spectrafolder)
      # Creation of spectra if they have not been created yet
      if not (f"{cat.df.name.values[i]}_spectrum.dat" in os.listdir(spectrafolder)):
        logE = np.logspace(1, 3, 100)  # energy (log scale)
        with open(spectrumfile, "w") as f:
          norm_val, spec, pht_pflx = norm_band_spec_calc(cat.df.alpha.values[i], cat.df.beta.values[i], cat.df.z_obs.values[i], cat.df.dl.values[i], cat.df.ep_rest.values[i], cat.df.liso.values[i], logE)
          f.write(f"#model normalized Band:   norm={norm_val}, alpha={cat.df.alpha.values[i]}, beta={cat.df.beta.values[i]}, epeak={cat.df.ep_rest.values[i]}keV\n")
          f.write(f"# Measured mean flux: {pht_mflx} ph/cm2/s in the 10-1000 keV band\n")
          f.write(f"# Measured peak flux: {pht_pflx} ph/cm2/s in the 10-1000 keV band\n")
          f.write("\nIP LOGLOG\n\n")
          for ite_E, E in enumerate(logE):
            f.write(f"DP {E} {spec[ite_E]}\n")
          f.write("\nEN\n\n")
      # Creation of the light curves files if not created yet
      if (not (f"LightCurve_{cat.df.name.values[i]}" in os.listdir("../Data/sources/Sample_Light_Curves")) or make_lc_files):
        gbm_cat = Catalog("../Data/CatData/allGBM.txt", [4, '\n', 5, '|', 4000], "../Data/CatData/rest_frame_properties.txt")
        closest_gmb_t90_name = cat.df.lc.values[i].split(".")[0].split("_")[-1]
        gbmt90 = gbm_cat.df[gbm_cat.df.name == closest_gmb_t90_name].t90.values[0]
        make_sample_lc(cat, i, gbmt90)
    else:
      spectrumfile = f"{args.spectrafilepath}{cat.df.name.values[i]}_spectrum.dat"
      lc_name = None
      model = cat.df.flnc_best_fitting_model.values[i]
      pht_mflx = cat.df.mean_flux.values[i]
      # pfluxmodel = cat.df.pflx_best_fitting_model.values[i]
      pht_pflx = cat.df.peak_flux.values[i]
      if np.isnan(pht_pflx):
        pht_pflx = "No value fitted"
      # if type(pfluxmodel) == str:
      #   pht_pflx = cat.df[f"{pfluxmodel}_phtflux"].values[i]
      # else:
      #   if np.isnan(pfluxmodel):
      #     pht_pflx = "No value fitted"
      #   else:
      #     raise ValueError("A value for pflx_best_fitting_model is not set properly")
      # Creation of spectra if they have not been created yet
      if not (f"{cat.df.name.values[i]}_spectrum.dat" in os.listdir(args.spectrafilepath)):
        logE = np.logspace(1, 3, 100)  # energy (log scale)
        with open(spectrumfile, "w") as f:
          f.write(f"#model {model}:  ")
          if model == "flnc_plaw":
            func_args = (cat.df.flnc_plaw_ampl.values[i], cat.df.flnc_plaw_index.values[i], cat.df.flnc_plaw_pivot.values[i])
            fun = plaw
            f.write(f"ampl={func_args[0]}, index={func_args[1]}, pivot={func_args[2]}keV\n")
          elif model == "flnc_comp":
            func_args = (cat.df.flnc_comp_ampl.values[i], cat.df.flnc_comp_index.values[i], cat.df.flnc_comp_epeak.values[i], cat.df.flnc_comp_pivot.values[i])
            fun = comp
            f.write(f"ampl={func_args[0]}, index={func_args[1]}, epeak={func_args[2]}keV, pivot={func_args[3]}keV\n")
          elif model == "flnc_band":
            func_args = (cat.df.flnc_band_ampl.values[i], cat.df.flnc_band_alpha.values[i], cat.df.flnc_band_beta.values[i], cat.df.flnc_band_epeak.values[i])
            fun = band
            f.write(f"ampl={func_args[0]}, alpha={func_args[1]}, beta={func_args[2]}, epeak={func_args[3]}keV\n")
          elif model == "flnc_sbpl":
            func_args = (cat.df.flnc_sbpl_ampl.values[i], cat.df.flnc_sbpl_indx1.values[i], cat.df.flnc_sbpl_indx2.values[i], cat.df.flnc_sbpl_brken.values[i], cat.df.flnc_sbpl_brksc.values[i], cat.df.flnc_sbpl_pivot.values[i])
            fun = sbpl
            f.write(f"ampl={func_args[0]}, index1={func_args[1]}, index2={func_args[2]}, eb={func_args[3]}keV, brksc={func_args[4]}keV, pivot={func_args[5]}keV\n")
          else:
            vprint(f"Could not find best fit model for {cat.df.name.values[i]} (indicated {model}). Aborting this GRB.", __verbose__, 0)
            return
          f.write(f"# Measured mean flux: {pht_mflx} ph/cm2/s in the 10-1000 keV band\n")
          f.write(f"# Measured peak flux: {pht_pflx} ph/cm2/s in the 10-1000 keV band\n")
          f.write("\nIP LOGLOG\n\n")
          for E in logE:
            f.write(f"DP {E} {fun(E, *func_args)}\n")
          f.write("\nEN\n\n")
    # Used for the GBM bursts that do not have pflux values, to still simulate them
    if pht_pflx == "No value fitted":
      pht_pflx = 1
    for j in range(args.simulationsperevent):
      dec_grb_world_frame, ra_grb_world_frame = random_grb_dec_ra(args.position[0], args.position[1], args.position[2], args.position[3])  # deg
      rand_time = np.around(np.random.rand()*315567360.0, 4)  # Time of the GRB, taken randomly over a 10 years time window
      for k, s in enumerate(args.satellites):
        orbital_period = orbital_period_calc(s[3])
        earth_ra_offset = earth_rotation_offset(rand_time)
        true_anomaly = true_anomaly_calc(rand_time, orbital_period)
        dec_sat_world_frame, ra_sat_world_frame = orbitalparam2decra(s[0], s[1], s[2], nu=true_anomaly)  # deg
        ra_sat_world_frame = np.mod(ra_sat_world_frame - earth_ra_offset, 360)
        if pht_pflx <= 0.1:
          save_log(f"{sim_directory}/simulation_logs.txt", cat.df.name.values[i], i, j, k, "Ignored(faint)", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, None, None)
        elif verif_rad_belts(dec_sat_world_frame, ra_sat_world_frame, s[3]):  # checks if sat is in the switch off zone
          save_log(f"{sim_directory}/simulation_logs.txt", cat.df.name.values[i], i, j, k, "Ignored(off)", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, None, None)
        else:
          theta, phi, theta_err, phi_err, thetap, phip, polstr = grb_decrapol_worldf2satf(dec_grb_world_frame, ra_grb_world_frame, dec_sat_world_frame, ra_sat_world_frame)[1:]
          if theta >= horizon_angle(s[3]):  # Source below horizon
            save_log(f"{sim_directory}/simulation_logs.txt", cat.df.name.values[i], i, j, k, "Ignored(horizon)", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
          else:
            # Add command to commands list
            if args.simtime.isdigit():
              simtime = float(args.simtime)
              lc_bool = False
            elif args.simtime == "t90":
              simtime = cat.df.t90.values[i]
              lc_bool = False
            elif args.simtime == "lc":
              simtime = cat.df.t90.values[i]
              lc_bool = True
            else:
              simtime = None
              lc_bool = False
              vprint("simtime in parameter file unknown. Check parameter file.", __verbose__, 0)
            args.commands.append((not(args.nocosima), not(args.norevan), not(args.nomimrec), cat.df.name.values[i], k, spectrumfile, pht_mflx, simtime, lc_bool, lc_name, polstr, j, f"{dec_grb_world_frame:.4f}_{ra_grb_world_frame:.4f}_{rand_time:.4f}", theta, phi))
            save_log(f"{sim_directory}/simulation_logs.txt", cat.df.name.values[i], i, j, k, "Simulated", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
  for i in range(len(cat.df)):
    gen_grb(i)
  with open(f"{sim_directory}/cosima_errlog.txt", "w") as errfile:
    vprint("Cosima error file created", __verbose__, 0)
  with open(f"{sim_directory}/revan_errlog.txt", "w") as errfile:
    vprint("Revan error file created", __verbose__, 0)
  with open(f"{sim_directory}/mimrec_errlog.txt", "w") as errfile:
    vprint("Mimrec error file created", __verbose__, 0)
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
  fname = f"tmp_{pid}_{command[3]}.source"
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
          if command[9] is None:  # Case using GBM data, catalog doesn't have light curve name so we find it using the GRB name
            f.write(f"\n{source}.Lightcurve File true ../Data/sources/GBM_Light_Curves/LightCurve_{command[3]}.dat")
          else:  # Case using Sampled data, catalog has light curve name so use it directly
            f.write(f"\n{source}.Lightcurve File true ../Data/sources/Sample_Light_Curves/{command[9]}")
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
  simfile, trafile, extrfile = f"{simname}.inc1.id1.sim.gz", f"{simname}.inc1.id1.tra.gz", f"{simname}.inc1.id1.extracted.tra"
  # mv_simname = f"{simname.split('/sim/')[0]}/rawsim/{simname.split('/sim/')[-1]}"
  # mv_simfile, mv_trafile = f"{mv_simname}.inc1.id1.sim.gz", f"{mv_simname}.inc1.id1.tra.gz"
  source_name = maketmpsf(command, args, pid)
  if command[0]:
    # Running cosima
    # if command[3] in ["GRB080804456", "GRB120420858", "GRB130215063", "GRB140603476"]:
    #   print(f"RUNNING {command[3]}")
    #   run(f"cosima -z {source_name}", 3)
    # else:
    run(f"cosima -z {source_name}; rm -f {source_name}", f"{simname.split('/sim/')[0]}/cosima_errlog.txt", simfile, __verbose__)
  if command[1]:
    # Running revan and moving the simulation file to rawsim
    # run(f"revan -g {args.geometry} -c {args.rcf} -f {simfile} -n -a; mv {simfile} {mv_simfile}", __verbose__)
    # Running revan and removing the simulation file
    run(f"revan -g {args.geometry} -c {args.rcf} -f {simfile} -n -a; rm -f {simfile}", f"{simname.split('/sim/')[0]}/revan_errlog.txt", trafile, __verbose__)
  if command[2]:
    # Running mimrec and moving the revan analyzed file to rawsim
    # run(f"mimrec -g {args.geometry} -c {args.mcf} -f {trafile} -x -n; mv {trafile} {mv_trafile}", __verbose__)
    # Running mimrec and removing the revan analyzed file
    run(f"mimrec -g {args.geometry} -c {args.mcf} -f {trafile} -x -n; rm -f {trafile}", f"{simname.split('/sim/')[0]}/mimrec_errlog.txt", extrfile, __verbose__)


def run(command, error_file, expected_file, __verbose__):
  """
  Runs a command
  :param command: str, shell command to run
  :param error_file: str, name of the error logfile
  :param __verbose__: verbosity
    0 -> No output
    1 -> One-line output (proscesses id, command and verbosity)
    2 -> Adds stderr of command
    3 -> Adds stdout of command
  """
  vprint(f"Process id {os.getpid()} from {os.getppid()} runs {command} (verbosity {__verbose__})", __verbose__, 0)
  if __verbose__ < 2:
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
    print(f"Script version {__version__} written by {__author__}.")
  if args.parameterfile:
    __verbose__ = args.verbose
    vprint(f"Running of {args.parameterfile} parameter file with output prefix {args.prefix}", __verbose__, 0)
    args = gen_commands(args)
    vprint(f"{len(args.commands)} Commands have been parsed", __verbose__, 0)
    run_sims(args.commands)
  else:
    vprint("Missing parameter file or geometry - not running.", __verbose__, 0)



