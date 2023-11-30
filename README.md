# grb-COMCUBE-simu

To be updated

requirements : astropy, cartopy

main folder contains all the codes
bkg contains information about the background, to be done
cfgs contains the configuration files for revan and mimrec
GBM contains the GBM data
geom contains the geometries 
mu100 will be containing values for mu100 at different position in the detector FoV
sources is made to contain the spectra of the sources simulated to be done

To run a simulation : 
create a folder in which everything will be stored
in this folder put :
  a param file "polGBM.par" (# EXAMPLES NEEDED#)
  a source file "wobkgGRB_Pol.source" (# EXAMPLES NEEDED#) 
  the mimrecAutoMT.py file
  create folders sim and rawsim
