# grb-COMCUBE-simu for the COMCUBE-S project

This repository was developed in the context of an ESA project. It aims at estimating the polarimetric and detection capabilities of the European COMCUBE-S satellite constellation. It uses Monte Carlo simulations and runs in a Python environment.

---

## Table of contents

1. [Overview](#overview)
2. [Project structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [References](#references)
7. [Authors](#authors)

---

## Overview

This repository serves several purposes:
- Computes the polarisation properties according to different astrophysical models (SO, SR, CD, PJ, see Toma et al. 2009 and Pearce et al. 2019)
- Creates GBM light curves
- Creates a synthetic GRB population catalogue
- Runs, transforms and analyses MEGALib Monte Carlo simulations for GRBs and background

---

## Project structure

```
grb-COMCUBE-simu/  ← project's repository 
├── src/
│   ├── Analysis/
│   │   ├── MAllSourceData.py                    # Container for a full set of GRB simulations and analysis methods
│   │   ├── MAllSimData.py                       # Container for all simulations of a given source and analysis methods
│   │   ├── MAllSatData.py                       # Container for one simulation of a given source (for all satellites) and analysis methods
│   │   ├── MGRBFullData.py                      # Data container for data at the satellite level
│   │   ├── MConstData.py                        # Data container for data at the constellation level
│   │   ├── MLogData.py                          # Reads and stores information from simulation log files 
│   │   ├── MBkgContainer.py                     # Container for background data
│   │   ├── MmuSeffContainer.py                  # Container for mu100 and effective area data 
│   │   ├── MFit.py                              # Container for fits
│   │   ├── find_detector.cxx                    # MEGAlib standalone to retrieve the detector where interactions occurred
│   │   └── Makefile                             # Makefile for find_detector.cxx
│   ├── Background/
│   │   ├── AlbedoPhotonBeam.dat                 # MEGAlib file to describe angular dependence of albedo photons
│   │   ├── BackgroundPlotter_All.py             # Plots all the different components.
│   │   ├── CreateBackgroundSpectrumMEGAlib.py   # Creates a file describing the spectrum of different components to be used with MEGAlib to define a source.
│   │   ├── LATBackground.py                     # Creates the file Data/LATBackground.dat from the Fermi fits file.
│   │   └── LEOBackgroundGenerator.py            # Contains the definition of the class describing all the background components.
│   ├── Catalogs/
│   │   ├── catalog.py                           # Contains functions and classes to read and contain data from GRB catalogues
│   │   ├── catalogMC.py                         # Synthetic GRB catalog generator (MCCatalog)
│   │   └── lightcurve_maker.py                  # GBM light curve downloader and builder
│   ├── Display/
│   │   └── visualisation.py                     # Contains various functions to visualise and calculate quantities independent of GRB simulations
│   ├── General/
│   │   └── funcmod.py                           # Contains core utility functions used in the project
│   ├── Launchers/
│   │   ├── launch_bkg_sim.py                    # Background simulation runner
│   │   ├── launch_mu100_sim.py                  # mu100 and effective area simulation runner
│   │   └── launch_sim_time.py                   # GRB simulation runner
│   ├── Polarization/
│   │   ├── models.py                            # Polarization fraction models (SO, SR, CD, PJ)
│   │   └── polarization_class.py                # PolVSAngleRatio runner
├── README.md
└── requirements.txt

Data/ ← External folder, to be downloaded separately
```
> **Note:** The Data/ folder is not part of the repository and must be downloaded separately (see the [Data](#data) section). It should be placed at the same level as the cloned repository, as the code expects relative paths of the form ../Data/.

---

## Data

Most of the code requires the `Data/` folder to function. Its structure is as follows:

```
Data/
├── bkg/        # Background simulations and information
├── cfgs/       # MEGAlib configuration files
├── catData/    # GRB catalogues 
├── geom/       # Instrument mass model (geometry files)
├── mu100/      # mu100/effective area simulations
├── sources/    # Source light curves and spectra
└── example/    # Example files
```

The `Data/` folder is not included in this repository due to its size.
It can be downloaded at: **https://zenodo.org/records/18998501**

### Condensed file naming conventions

Condensed files aggregate simulation results and are required for analysis without raw simulation data. They are specific to the energy cut applied during their creation — using a different energy cut will produce incorrect results.

| Type | Regular filename | Condensed filename |
|------|------------------|--------------------|
| Background | `[prefix]_[model]_[decmin]-[decmax]-[ndec]_[altmin]-[altmax]-[nalt].txt` | `..._ergcut-[emin]-[emax].txt` |
| mu100 / Seff | `[prefix]_[model]_[decmin]-[decmax]-[ndec]_[ramin]-[ramax]-[nra].txt` | `..._ergcut-[emin]-[emax].txt` |

Some pre-computed condensed files for background and mu100 are already included in `Data/`. They allow analysis to be performed even without raw simulation data, provided the energy cut matches.

---

## Requirements

### Python environment

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

or using conda (recommended for `cartopy` and `pyside2`):

```bash
conda install numpy matplotlib scipy pandas astropy cartopy pyqt pyside2
pip install apexpy
```

See `requirements.txt` for the full list of dependencies.

### GBM Data Tools (separate environment)

The GBM Data Tools package has dependency conflicts with the main environment. **It must be installed in a dedicated virtual or conda environment.**

1. Download the installation package from: https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/
2. Install with pip (conda is not supported):
   ```bash
   pip install <path_to_tar>/gbm_data_tools-1.1.1.tar.gz
   ```
   > If the archive was auto-decompressed during download but still has a `.gz` extension, remove the `.gz` suffix before running the command.
   
### Other dependencies

- `make` (for compiling `find_detector.cxx`)
- MEGAlib — must be installed **and sourced** before running any simulation

---

## Installation
```bash
git clone https://github.com/nfranel/grb-COMCUBE-simu
cd grb-COMCUBE-simu
```
And create your Python environment using the commands given previously in [Python environment](#python-environment)

---

## Usage

### Before running simulations

Make sure the following are in place for each satellite geometry:

- Geometry / mass model file (`.geo.setup`)
- MEGAlib configuration files for revan and mimrec (`.cfg`)
- A condensed background file for the target geometry and energy cut
- A condensed mu100 / Seff file for the target geometry and energy cut
- For each GRB source, a simulation folder containing:
  - a source file (`.source`)
  - a parameter file
  - a `sim/` subfolder
  - a `rawsim/` subfolder

> **Note:** By default, simulation runners delete raw and revan-analysed files after processing to save disk space; this behaviour can be changed inside the launcher scripts.

---

### Simulation runners

#### GRB simulations
```bash
python launch_sim_time.py -f grb_param_file
```

#### Background simulations
```bash
python launch_bkg_sim.py -f bkg_param_file
```

#### mu100 / Seff simulations
```bash
python launch_mu100_sim.py -f mu100_param_file
```

---

### Download and build GBM light curves

Run in the dedicated GBM Data Tools environment:

```bash
python lightcurve_maker.py
```

---

### Generate a synthetic GRB catalogue

```python
from src.Catalogs.catalogMC import MCCatalog

testcat = MCCatalog(mode="catalog")      # Generate one or more accepted catalogues
testcat = MCCatalog(mode="mc")           # Explore the parameter space (Monte Carlo)
testcat = MCCatalog(mode="parametrized") # Run with a fixed parameter set
```

---

### Compute polarization fractions

```python
from src.Polarization.polarization_class import PolVSAngleRatio

model = "SO"  # or "SR", "CD", "PJ"
distribution = PolVSAngleRatio(model=model)
distribution.pf_calculation()
```

---

## References

- Thesis about this work
  - Nathan Franel 2025, https://theses.hal.science/tel-05401994 
    *See this thesis for available arXiv links and further information*
- MEGAlib software
  - A. Zoglauer, R. Andritschke, and F. Schopper, https://doi.org/10.1016/j.newar.2006.06.049
- Polarisation models
  - K. Toma et al. 2009, https://doi.org/10.1088/0004-637X/698/2/1042
  - M. Pearce et al. 2019, https://doi.org/10.1016/j.astropartphys.2018.08.007
- Fermi GBM GRB catalogue
  - S. Poolakkil et al. 2021, https://doi.org/10.3847/1538-4357/abf24d
- Synthetic catalog creation
  - D. Band et al. 1993, https://doi.org/10.1086/172995
  - G. Ghirlanda et al. 2015, https://doi.org/10.1016/j.jheap.2015.04.002
  - G. Ghirlanda et al. 2016, https://doi.org/10.1051/0004-6361/201628993
  - G. Ghirlanda and R. Salvaterra 2022, https://doi.org/10.48550/arXiv.2206.06390
  - A. Lien et al. 2014, https://doi.org/10.1088/0004-637X/783/1/24
  - J. Palmerio et al. 2021, https://doi.org/10.1051/0004-6361/202039929
  - D. Wanderman & T. Piran 2010, https://doi.org/10.1111/j.1365-2966.2010.16787.x
  - D. Yonetoku et al. 2004, https://doi.org/10.1086/421285
- Background sources
  - P. Cumani et al. 2019, https://doi.org/10.1007/s10686-019-09624-0

---

## Authors

- Nathan Franel
- Adrien Laviron
