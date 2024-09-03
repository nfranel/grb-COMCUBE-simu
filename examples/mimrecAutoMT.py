"""
**************************************************************************
*                                                                        *
*            Mimrec - MEGAlib image reconstruction (and more)            *
*                                                                        *
*             This program is part of MEGAlib version 3.03.01            *
*                (C) by Andreas Zoglauer and contributors                *
*                                                                        *
*                      Master reference for MEGAlib:                     *
*            A. Zoglauer et al., NewAR 50 (7-8), 629-632, 2006           *
*                                                                        *
*            For more information about MEGAlib please visit:            *
*                        http://megalibtoolkit.com                       *
*                                                                        *
**************************************************************************

              You are using a development version of MEGAlib              


  Usage: mimrec <options>

    Basic options:
      -g --geometry <filename>.geo.setup:
             Use this file as geometry-file
      -f --filename <filename>.tra:
             This is the file which is going to be analyzed
      -c --configuration <filename>.cfg:
             Use this file as parameter file (uses files from -f and -g)
             If no configuration file is give ~/.mimrec.cfg is used
      -C --change-configuration <pattern>:
             Replace any value in the configuration file (-C can be used multiple times)
             E.g. to replace the standard ARM cut range with 10 degrees, one would set pattern to:
             -C TestPositions.DistanceTrans=10
      -d --debug:
             Use debug mode
      -h --help:
             You know the answer...

    High level functions:
      -o --output:
             For --image, --spectrum, --light-curve, --distances, --scatter-angles, --sequence-length, and --arm-gamma: Save the generated histogram.
             For -x: Save the extracted events
             If multiple histograms are generated, additional modifiers will be added to the file name
      -i --image:
             Create an image. If the -o option is given then the image is saved to this file.
      -s --spectrum:
             Create a spectrum. If the -o option is given then the image is saved to this file.
      -a --arm-gamma:
             Create an arm. If the -o option is given then the image is saved to this file.
      -l --light-curve:
             Create a light curve. If the -o option is given then the image is saved to this file.
      -p --polarization:
             Perform polarization analysis. If the -o option is given then the image is saved to this file.
         --interaction-distance:
             Create interaction distance plots. If the -o option is given then the image is saved to this file.
         --scatter-angles:
             Create the scatter-angle distributions. If the -o option is given then the image is saved to this file.
         --sequence-length:
             Create the sequence length plots. If the -o option is given then the image is saved to this file.
      -x --extract:
             Extract events using the given event selection to the file given by -o
      -e --event-selections:
             Dump event selections
         --standard-analysis-spherical <energy [keV]> <theta [deg]> <phi [deg]>
             Do a standard analysis (Spectra, ARM, Counts) and dump the results to a *.sta file

    Additional options for high level functions:
      -n --no-gui:
             Do not use a graphical user interface
      -k --keep-alive:
             Do not quit after executing a batch run, if we do have a gui
"""


import multiprocessing as mp
import subprocess
import glob
import time

# =============================================================================== #
# The potential changes in geometry and other files are done there
# =============================================================================== #
repository_path = "/absolute/path"
geometry = "file_name_of_geometry"
mimrecfile = "file_name_of_mimrec_file"
sim_folder = "./sim/"
rawsim_folder = "./rawsim/"


def genCommands(path):
  """
  Generate all commands to extract events with mimrec from multiple files
  :param path: path of .tra.gz files
  :returns: list of str
  """
  coms = []
  geom = f"{repository_path}/grb-COMCUBE-simu/COMCUBE_M7/{geometry}"
  mimfile = f"{repository_path}/grb-COMCUBE-simu/cfgs/{mimrecfile}"
  if not (path.endswith("/")):
    path += "/"
  # flist = subprocess.getoutput("ls {}".format(path)).split("\n")
  flist = glob.glob(f"{path}")
  for f in flist:
    coms.append("mimrec -g {0} -c {1} -f {2}{3} -x -n; mv {2}{3} {4}".format(geom, mimfile, path, f, rawsim_folder))
  return coms


def runCommand(com):
  """
  Runs single command com
  :param com: str
  """
  print(com)
  subprocess.call(com, shell=True)


def runMT(commands):
  """
  Runs all commands in commands with multi-threading
  :param commands: list of str
  """
  with mp.Pool() as pool:
    pool.map(runCommand, commands)


if __name__ == "__main__":
  commands = genCommands(sim_folder)
  for t in range(10, 0, -1):
    print("Generated {} commands. Abort with Ctrl-C before {} s.".format(len(commands), t), end="\r")
    time.sleep(1)
  runMT(commands)


































