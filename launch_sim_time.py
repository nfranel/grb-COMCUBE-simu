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
# Developped modules imports
from funcmod import *
from catalog import Catalog

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
  args.geometry, args.rcf, args.mcf, args.spectrafilepath, args.grbfile, args.csf, args.prefix, args.sttype, args.simulationsperevent, args.simtime, args.position, args.satellites = read_grbpar(args.parameterfile)
  c = Catalog(args.grbfile, args.sttype)
  vprint("Running with GBM data on flnc mode.", __verbose__, 0)
  items = "flnc_plaw_ampl,flnc_plaw_pivot,flnc_plaw_index,flnc_plaw_phtflux,flnc_comp_ampl,flnc_comp_epeak,flnc_comp_index,flnc_comp_pivot,flnc_comp_phtflux,flnc_band_ampl,flnc_band_epeak,flnc_band_alpha,flnc_band_beta,flnc_band_phtflux,flnc_sbpl_ampl,flnc_sbpl_pivot,flnc_sbpl_indx1,flnc_sbpl_brken,flnc_sbpl_brksc,flnc_sbpl_indx2,flnc_sbpl_phtflux".split(",")
  defaults = [0, 100, 0, 0, 0, 1, 0, 100, 0, 0, 1, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0]
  c.tofloats(items, defaults)
  args.commands = []
  sim_directory = args.prefix.split("/sim/")[0]
  with open(f"{sim_directory}/simulation_logs.txt", "w") as f:
    f.write("========================================================================\n")
    f.write("                    Log file for the simulations                        \n")
    f.write("GRB name | simulation number | satellite number | status of the simulation | sat inclination | sat RA of ascending node | sat argument of periapsis | altitude | random time of simulation | sat dec world frame | sat ra world frame | grb dec world frame | grb ra world frame | grb dec sat frame | grb ra sat frame\n")
    f.write("name | sim_num | sat_num | status | inc | ohm | omega | alt | rand_time | sat_decwf | sat_rawf | grb_decwf | grb_rawf | grb_decsf | grb_rasf")
    f.write("Angles in degrees and altitude in km\n")
    f.write("========================================================================\n")
  def gen_grb(i): #Generate a single GRB command
    #generate spectrum file here if not already done
    if not(args.spectrafilepath.endswith("/")) and os.name == "posix": args.spectrafilepath += "/"
    spectrumfile = "{}{}_spectrum.dat".format(args.spectrafilepath, c.name[i])
    model = getattr(c, "flnc_best_fitting_model")[i].strip()
    pfluxmodel = getattr(c, 'pflx_best_fitting_model')[i].strip()
    pht_mflx = getattr(c, f"{model}_phtflux")[i]
    pht_pflx = getattr(c, f"{pfluxmodel}_phtflux")[i]
    if not(spectrumfile in os.listdir(args.spectrafilepath)):
      logE = np.logspace(1, 3, 100)#energy (log scale)
      with open(spectrumfile, "w") as f:
        f.write("#model {}:  ".format(model))
        # Commented as we do not simulate grbs with the peak flux
        # if model == "pflx_plaw":
        #   fun = lambda x: plaw(x, c.pflx_plaw_ampl[i], c.pflx_plaw_index[i], c.pflx_plaw_pivot[i])
        #   f.write("ampl={}, index={}, pivot={}keV\n".format(c.pflx_plaw_ampl[i], c.pflx_plaw_index[i], c.pflx_plaw_pivot[i]))
        # elif model == "pflx_comp":
        #   fun = lambda x: comp(x, c.pflx_comp_ampl[i], c.pflx_comp_index[i], c.pflx_comp_epeak[i], c.pflx_comp_pivot[i])
        #   f.write("ampl={}, index={}, epeak={}keV, pivot={}keV\n".format(c.pflx_comp_ampl[i], c.pflx_comp_index[i], c.pflx_comp_epeak[i], c.pflx_comp_pivot[i]))
        # elif model == "pflx_band":
        #   fun = lambda x: band(x, c.pflx_band_ampl[i], c.pflx_band_alpha[i], c.pflx_band_beta[i], c.pflx_band_epeak[i])
        #   f.write("ampl={}, alpha={}, beta={}, epeak={}keV\n".format(c.pflx_band_ampl[i], c.pflx_band_alpha[i], c.pflx_band_beta[i], c.pflx_band_epeak[i]))
        # elif model == "pflx_sbpl":
        #   fun = lambda x: sbpl(x, c.pflx_sbpl_ampl[i], c.pflx_sbpl_indx1[i], c.pflx_sbpl_indx2[i], c.pflx_sbpl_brken[i], c.pflx_sbpl_brksc[i], c.pflx_sbpl_pivot[i])
        #   f.write("ampl={}, index1={}, index2={}, eb={}keV, brksc={}keV, pivot={}keV\n".format(c.pflx_sbpl_ampl[i], c.pflx_sbpl_indx1[i], c.pflx_sbpl_indx2[i], c.pflx_sbpl_brken[i], c.pflx_sbpl_brksc[i], c.pflx_sbpl_pivot[i]))
        if model == "flnc_plaw":
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
      dec_grb_world_frame, ra_grb_world_frame = random_GRB_dec_ra(args.position[0], args.position[1], args.position[2], args.position[3])  # deg
      rand_time = np.around(np.random.rand()*315567360.0, 4) # Time of the GRB, taken randomly over a 10 years time window
      for k, s in enumerate(args.satellites):
        orbital_period = orbital_period_calc(s[3])
        earth_ra_offset = earth_rotation_offset(rand_time)
        true_anomaly = true_anomaly_calc(rand_time, orbital_period)
        dec_sat_world_frame, ra_sat_world_frame = orbitalparam2decra(s[0], s[1], s[2], nu=true_anomaly)  # deg
        ra_sat_world_frame -= earth_ra_offset
        # if verif_zone(90 - dec_sat_world_frame, ra_sat_world_frame):  # checks if the sat is in the switch off zone
        if verif_rad_belts(90 - dec_sat_world_frame, ra_sat_world_frame, s[3]):  # checks if the sat is in the switch off zone
          save_log(f"{sim_directory}/simulation_logs.txt", c.name[i], j, k, "Ignored(off)", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, None, None)
        else:
          theta, phi, thetap, phip, polstr = grb_decrapol_worldf2satf(dec_grb_world_frame, ra_grb_world_frame, dec_sat_world_frame, ra_sat_world_frame)[1:]
          if theta >= horizonAngle(s[3]):#source below horizon
            save_log(f"{sim_directory}/simulation_logs.txt", c.name[i], j, k, "Ignored(horizon)", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
          else:
            # Add command to commands list
            if args.simtime.isdigit():
              simtime = float(args.simtime)
            elif args.simtime == "t90":
              simtime = float(c.t90[i])
            else:
              simtime = None
              vprint("simtime in parameter file unknown. Check parameter file.", __verbose__, 0)
            args.commands.append((not(args.nocosima), not(args.norevan), not(args.nomimrec), c.name[i], k, spectrumfile, pht_mflx, simtime, polstr, j, f"{dec_grb_world_frame:.1f}_{ra_grb_world_frame:.1f}_{rand_time:.4f}", theta, phi))
            save_log(f"{sim_directory}/simulation_logs.txt", c.name[i], j, k, "Simulated", s[0], s[1], s[2], s[3], rand_time, dec_sat_world_frame, ra_sat_world_frame, dec_grb_world_frame, ra_grb_world_frame, theta, phi)
  for i in range(len(c)):
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
    run, source = "", "" # So that the test with startswith may be done even if run and source are not initialized yet
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
      elif line.startswith(f"{source}.Beam") and (source=="GRBsource" or source=="GRBsourcenp"):
        f.write(f"{source}.Beam FarFieldPointSource {command[-2]} {command[-1]}")
      elif line.startswith(f"{source}.Spectrum") and (source=="GRBsource" or source=="GRBsourcenp"):
        f.write(f"{source}.Spectrum File {command[5]}")
      elif line.startswith(f"{source}.Polarization") and (source=="GRBsource" or source=="GRBsourcenp"):
        f.write(f"{source}.Polarization Absolute 1. {command[8]}")
      elif line.startswith(f"{source}.Flux") and (source=="GRBsource" or source=="GRBsourcenp"):
        f.write(f"{source}.Flux {command[6]}")
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
    print("Running : f'cosima -z {source_name}; rm -f {source_name}'")
    run(f"cosima -z {source_name}; rm -f {source_name}", __verbose__)
    subprocess.call("ll /pdisk/ESA/400km--0-0-0--27sat/sim")
    subprocess.call("ll /pdisk/ESA/400km--0-0-0--27sat/rawsim")
  if command[1]: #run revan
    # Running revan
    print("Running : f'revan -g {args.geometry} -c {args.rcf} -f {simfile} -n -a; mv {simfile} {mv_simfile}'")
    run(f"revan -g {args.geometry} -c {args.rcf} -f {simfile} -n -a; mv {simfile} {mv_simfile}", __verbose__)
    subprocess.call("ll /pdisk/ESA/400km--0-0-0--27sat/sim")
    subprocess.call("ll /pdisk/ESA/400km--0-0-0--27sat/rawsim")
    # run(f"revan -g {args.geometry} -c {args.rcf} -f {simfile} -n -a; rm -f {simfile}", __verbose__)
  if command[2]:
    # Running mimrec
    print("Running : f'mimrec -g {args.geometry} -c {args.mcf} -f {trafile} -x -n; mv {trafile} {mv_trafile}'")
    run(f"mimrec -g {args.geometry} -c {args.mcf} -f {trafile} -x -n; mv {trafile} {mv_trafile}", __verbose__)
    subprocess.call("ll /pdisk/ESA/400km--0-0-0--27sat/sim")
    subprocess.call("ll /pdisk/ESA/400km--0-0-0--27sat/rawsim")
    # run(f"mimrec -g {args.geometry} -c {args.mcf} -f {trafile} -n -a; rm -f {trafile}", __verbose__)


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



