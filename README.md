# grb-COMCUBE-simu

requirements : astropy, cartopy

############################################################################
This git contains :

- the codes and routines to analyze the data : shape M....py
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

Geometry
Background condensed file for specific geometry
Mu100/Seff condensed file for specific geometry
cfg fileq
A folder to contain the simulation with :
  source file
  parameter file
  a folder named sim
  a folder named rawsim

############################################################################
To run the simulations :
  Make sure the necessary files and folders are created
  use python specific_launcher -f specific_parameter_file

For background simulations : launch_bkg_sim with param file in bkg folder
For mu100 simulations : launch_mu100_sim with param file in mu100 folder
For GRB simulations : launch_sim_time with param file in the folder to contain simulation
  launch_sim is the old version, not up to date

############################################################################
To analyze results :
The maintest.py file gives an example of how to load data