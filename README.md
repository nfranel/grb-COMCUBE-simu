# grb-COMCUBE-simu

requirements : astropy, cartopy

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
  - contains information about the background, to be done 
- the folder cfgs 
  - contains the configuration files for revan and mimrec 
- the folder GBM 
  - contains the GBM data 
- the folder geom 
  - contains the geometries 
- the folder mu100 will be 
  - containing values for mu100 at different position in the detector FoV 
- the folder sources is made to 
  - contain the spectra of the sources simulated to be done

To run a simulation : 
create a folder in which everything will be stored
in this folder put :
  a param file "polGBM.par" (# EXAMPLES NEEDED#)
  a source file "wobkgGRB_Pol.source" (# EXAMPLES NEEDED#) 
  the mimrecAutoMT.py file
  create folders sim and rawsim
