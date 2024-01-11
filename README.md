# grb-COMCUBE-simu

Python requirements :
  - numpy, matplotlib, scipy, pandas, argparse, multiprocessing, subprocess, time, os, gzip, inspect, itertools
  - Special requirements :
    - astropy, cartopy
    
Other requirements :
  - make
  - megalib installed and **sourced**

############################################################################

This git contains :

- the codes and routines to analyze the data :
  - MAllSourceData.py contains the main class for analysis. Used with python, it reads and analyzes the simulation files
    - stores the simulation results for each source in a list
    - stores mu100 and effective area for different DEC and RA (in satellite frame)
    - stores bkg data for different latitudes at which a GRB may be detected (depends on the observing satellite)
    - several useful information on the simulation
  - MAllSimData.py contains a class to contain source results for all simulations
    - stores the results of simulation for one source/GRB for all simulations
    - stores several useful information on the source
  - MAllSatData.py contains a class to contain results for 1 source and 1 simulation of this source for all satellites
    - stores the results of simulation for each sat and possibly for a constellation
    - stores several information relative to 1 simulation of 1 source 
  - MGRBFullData.py contains the data from the simulation files
    - stores the data from the simulation files for a specified energy cut and ARM cut
    - stores some results obtained from the data 
  - MLogData.py contains a class to create a logfile while running the simulations
    - Information on simulations (GRB position, run or not (due to exclusion zones or below the horizon), etc)
  - MBkgContainer.py contains classes to treat bkg data, save it for a quicker use and to store results for analysis
  - MmuSeffContainer.py contains classes to treat mu100 and effective area for different positions of detection
    - These variables need statistics to be consistent and are considered to vary over detection position but not much over different GRBs  
  - MFit.py contains a class to proceed to polarigram fits
  - funcmod.py contains many functions used in the codes
  - catalog.py is used to read and extract the GBM data
  - fovconst.py is a code to obtain the field of view of a constellation (May need some updates)
  - trajectories.py is used to obtain the trajectory of a constellation, test exclusion files and obtain duty cycles
  - maintest.py is a file to test the data analysis, contains an example of how to make an analysis 
  - Launchers :
    - launch_bkg_sim.py    simulate the background for different latitudes, altitudes and for a specific satellite geometry 
    - launch_mu100_sim.py  simulate the mu100 and effective area for different declination and right ascension of detection for a specific satellite geometry
    - launch_sim_time.py   simulate the GRB simulations
  - find_detector.cxx and Makefile are the c++ program used to obtain the detector of interaction for an event, and its associated makefile. This uses megalib classes and has been done using the standalone example given by megalib.
    - The makefile compiles this program everytime the analysis is launched using the command "make -f Makefile PRG=find_detector". The executable is created in $(MEGALIB)/bin and is usable using "find_detector" in a terminal if megalib is sourced.
    - This program uses megalib classes then it is necessary to have megalib sourced while using this analysis tool

- the folder bkg 
  - contains information about the background 
    - codes to simulate the background spectra and the data it uses in a folder
    - folder where the spectra are saved
    - parameter and source file to run the background simulations
    - folder with files containing the exclusion area where the satellite is switch off
    - folder containing the simulations (Empty if not simulations were made) for a specific geometry and different latitudes
      - contains a file with condensed data for saving and a quicker use in the analysis 
- the folder cfgs 
  - contains the configuration files for revan and mimrec 
- the folder GBM 
  - contains the GBM data for short and long bursts
- the folder geom 
  - contains the geometries 
- the folder mu100 will be 
  - parameter and source files
  - containing simulations for mu100 at different position in the detector FoV (Empty if not simulations were made) for a specific geometry
    - contains a file with condensed data for saving and a quicker use in the analysis
- the folder sources is made to 
  - contain the spectra of the sources simulated (best fit spectra obtained from GBM data)
  - contain the spectrum for a typical GRB (band spectrum)
- the folder example that contains some example files (parameter file, source file)

############################################################################

Necessary to run the simulation

- Geometry
- Background condensed file for specific geometry
- Mu100/Seff condensed file for specific geometry
- cfg files (for both revan and mimrec)
- A folder to contain the simulation with :
  - source file
  - parameter file
  - a folder named sim
  - a folder named rawsim

IMPORTANT : 
- mu100 and background simulation files have to be made for every satellite model.
- Some condensed files for background and mu100 are already done on the git. With the files the analysis can be done even if there are no raw simulation data
- Condensed files are also specific to an energy cut so if the energy cut is not the same as the one used for creating these files results may be wrong
  - Format for names of background saved files :
    - regular one : [prefix]_[model]_[decmin]-[decmax]-[number of dec]_[altmin]-[altmax]-[number of alts].txt
    - condensed one : [prefix]_[model]_[decmin]-[decmax]-[number of dec]_[altmin]-[altmax]-[number of alts]_ergcut-[low energy cut]-[high energy cut].txt
  - Format for names of mu100 saved files :
    - regular one : [prefix]_[model]_[decmin]-[decmax]-[number of dec]_[ramin]-[ramax]-[number of ra at equator].txt
    - condensed one : [prefix]_[model]_[decmin]-[decmax]-[number of dec]_[ramin]-[ramax]-[number of ra at equator]_ergcut-[low energy cut]-[high energy cut].txt
      

############################################################################

To run the simulations :
- Make sure the necessary files and folders are created
- use python specific_launcher -f specific_parameter_file


- For background simulations : 
  - launch_bkg_sim with param file in bkg folder

- For mu100 simulations : 
  - launch_mu100_sim with param file in mu100 folder

- For GRB simulations : 
  - launch_sim_time with param file in the folder to contain simulation

IMPORTANT : in these files there is a possibility to keep the raw simulation files and the revan analyzed files. By default we remove this file to save memory.

############################################################################

To analyze results :
- The maintest.py file gives an example of how to load data