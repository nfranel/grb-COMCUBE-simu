"""
*.tra file analysis utilities for GRB detection rate simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

import os
import subprocess
import gzip
import h5py

from catalog import *

### Physics functions

def inwindow(E, ergCut):
  """
  Checks whether E is in the energy window defined by ergCut
  :param E: energy
  :param ergCut: (Emin, Emax)
  :returns: bool
  """
  return np.logical_and(E > ergCut[0], E < ergCut[1])

def SNR(S, B, C=0):
  """
  Calculates the signal to noise ratio of a GRB in a time bin
  :param S: number of counts in the source (background included)
  :param B: expected number of background counts
  :param C: minimum number of counts in the source to consider the detection
  :returns: SNR (as defined in Sarah Antier's PhD thesis
  """
  return (S-B)/np.sqrt(B+C)

def MDP(S, B, mu100, nsigma=4.29):
  """
  Calculates the minimum detectable polarization for a burst
  :param S: number of expected counts from the burst
  :param B: number of expected counts from the background
  :param mu100: modulation factor
  :param nsigma: significance of the result in number of sigmas, default=4.29 for 99% CL
  """
  return nsigma*np.sqrt(S+B)/(mu100*S)

def horizonAngle(h, EarthRadius = 6371, AtmosphereHeight = 40):
  """
  Calculates the angle between the zenith and the horizon for a LEO satellite
  :param h: altitude of the satellite (km)
  :param EarthRadius: radius of the Earth (km), default=6371
  :param AtmosphereHeight: height of the atmosphere (km), default=40
  :returns: horizon angle (deg)
  """
  return 90 + np.rad2deg(np.arccos( (EarthRadius + AtmosphereHeight) / (EarthRadius + h) )) #deg

def plaw(e, A, l, pivot=100):
  """
  Power-law spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param l: spectral index
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  return A*(e/pivot)**l

def comp(e, A, l, ep, pivot=100):
  """
  Comptonized spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param l: spectral index
  :param ep: peak energy (keV)
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  return A*(e/pivot)**l*np.exp(-(l+2)*e/ep)

def glog(e, A, ec, s):
  """
  log10-gaussian spectrum model
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param ec: central energy (keV)
  :param s: distribution width
  :returns: ph/cm2/keV/s
  """
  return A/np.sqrt(2*np.pi*s)*np.exp(-.5*(np.log10(e/ec)/s)**2)

def band(e, A, alpha, beta, ep, pivot=100):
  """
  Band spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param alpha: low-energy spectral index
  :param beta: high-energy spectral index
  :param ep: peak energy (keV)
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  c = (alpha-beta)*ep/(alpha+2)
  if e > c:
    return A*(e/pivot)**beta*np.exp(beta-alpha)*(c/pivot)**(alpha-beta)
  else:
    return A*(e/pivot)**alpha*np.exp(-(alpha+2)*e/ep)

def sbpl_sa(e, A, l1, l2, eb, delta, pivot=100):
  """
  Smoothly broken power law spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  """
  b, m = .5*(l1+l2), .5*(l1-l2)
  q, qp = np.log10(e/eb/delta), np.log10(pivot/eb/delta)
  a, ap = m*delta*np.log(np.cosh(q)), m*delta*np.log(np.cosh(qp))
  return A*(e/pivot)**b*10**(a/ap)

def sbpl(e, A, l1, l2, eb, delta, pivot=100):
  """
  Smoothly broken power law spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  """
  b, m = .5*(l2+l1), .5*(l2-l1)
  q, qp = np.log10(e/eb)/delta, np.log10(pivot/eb)/delta
  a, ap = m*delta*np.log(np.cosh(q)), m*delta*np.log(np.cosh(qp))
  return A*(e/pivot)**b*10**(a-ap)

def skyScatterPlot(theta, phi, other=None, projection="aitoff", **kwargs):
  """
  Plots a list of positions on the sky
  :param theta: list of theta in [0, 180] deg
  :param phi: list of phi in [0, 360] deg
  :other: list of values to be shown on a colorbar, default=None
  :param projection: str, sky projection, 
  """
  plt.subplot(111, projection=projection)
  thetap = np.deg2rad(90-np.array(theta))
  phip = np.mod(np.deg2rad(np.array(phi))+np.pi, 2*np.pi)-np.pi
  if other is None:
    plt.scatter(phip, thetap, **kwargs)
  else:
    sc = plt.scatter(phip, thetap, c=other, **kwargs)
    plt.colorbar(sc)


########## CLASS LOADINGBAR ##########

class LoadingBar:

  def __init__(self, tot, nprint=0, n=40, smtg=""):
    """
    Initiates loading bar
    :param tot: scalar or iterable, total number of steps or object to iterate from
    :param n: int, number of cases used to represent the loading, default=40
    :param smtg: str, printed over the loading bar
    """
    print("Loading : "+smtg+"\n["+n*" "+"]", end='\r', flush=True)
    self.n = n
    if type(tot) == int or type(tot) == float:
      self.tot = tot
    else:
      self.range = tot
      self.now = 0
      self.tot = len(tot)
    if nprint == 0:
      self.nprint = int(self.tot/self.n)+1
    else:
      self.nprint = nprint

  def update(self, now):
    """
    Updates loading bar
    :param now: scalar, current step
    """
    if now % self.nprint == 0:
      print("["+int(self.n*now/self.tot+.5)*"#", end='\r', flush=True)

  def end(self):
    """
    Ends loading bar
    """
    print("["+self.n*"#"+"]")
  
  def __iter__(self):
    """
    Setup LoadingBar to be used as iterable
    """
    self.iterator = iter(self.range)
    return self

  def __next__(self):
    """
    Updates loading bar if used as iterable
    """
    self.update(self.now)
    self.now += 1
    return next(self.iterator)

########## CLASS TRAEVENT ##########

#class Traevent:
#  """
#  """
#  def __init__(self, dat):
#    """
#    """
#    if type(dat) == list and dat[0]=="SE":
#      self.evtType = dat[1].split(" ")[1]
#      self.id = int(dat[2][3:])
#      self.time = float(dat[3][3:])
#      if self.evtType == "CO":
#        pass
#      elif self.evtType == "PH":
#        self.erg = float(dat[4][3:])
#        self.pos = (float(e) for e in dat[5].split(" ")[1:])

########## CLASS TRALIST ##########

#class Tralist(list):
#  """
#  """
#  def __init__(self, dat):
#    """
#    """
#    pass
#
#  def countsInVol(self, xmin, xmax, ymin, ymax, zmin, zmax):
#    """
#    """
#    pass
  
########## CLASS TRAFILE ##########

class Trafile:
  """
  """
  def __init__(self, dat=None, items=["ET", "TI"], optanalysis=[None, float]):
    """
    Instanciate Trafile
    :param dat: str or iterable of len 7, name of file to read or values of the main fields
        - triggers: int
        - calor: int
        - dsssd: int
        - side: int
        - single: int
        - compton: int
        - name: int or str, simnumber or {prefix}_GRB{datenumber}-sat{number}-{simnumber}_{theta}_{phi}.inc*.id*.tra(.gz)
        - theta: declination in degrees
        - phi: right ascension in degrees
    :param items: list of items to look for in each event, also name of fields they will be stored in, default=["ET", "TI"]
    :param optanalysis: list of functions (or None) to perform on each item, default=[None, float]
    :returns: Trafile
    """
    self.triggers, self.calor, self.dsssd, self.side, self.single, self.compton = [0]*6
    if type(dat) == str:#Load file
      ## Ajout pour traiter le cas où aucun fichier n'est créé pour un satellite donné / probleme a priori resolu dans trafilelist
      #if dat.startswith("ls: cannot access"):
      #  self.name = None
      #  self.theta = None
      #  self.phi = None
      #  for item in items:
      #    setattr(self, item, [])
      #else:
      self.name = dat
      temp = dat.split("_")
      self.theta = float(temp[2])
      self.phi = float('.'.join(temp[3].split('.')[:2])) #float(temp[3][:9])
      if dat.endswith(".tra"):
        with open(dat) as f:
          lines = f.read().split("\n")
      elif dat.endswith(".tra.gz"):
        with gzip.open(dat, "rt") as f:
          lines = [e[:-1] for e in f]
      else:
        raise TypeError("{} has unknown extension (known: .tra ou .tra.gz)".format(dat))
      for item in items:
        setattr(self, item, [])
      for i, line in enumerate(lines):
        if line.startswith("  Number of triggered events:"):
          self.triggers = int(line.split(" ")[-1])
        elif line.startswith("      TCeBr3Det:"):
          self.calor = int(line.split(" ")[-1])
        elif line.startswith("      TSiSDDet:"):
          self.dsssd = int(line.split(" ")[-1])
        elif line.startswith("      TSideDetector:"):
          self.side = int(line.split(" ")[-1])
        elif line.startswith("       Single-site"):
          self.single = int(line[58:66])
        elif line.startswith("       Compton"):
          self.compton = int(line[58:66])
        for item in items:
          if line.startswith(item):
            #getattr(self, item).append(line.split(" ")[1:])
            getattr(self, item).append(line.split(" ")[1:] if len(line.split(" ")[1:]) > 1 else line.split(" ")[1])
      for item, f in zip(items, optanalysis):
        if f != None:
          setattr(self, item, list(map(f, getattr(self, item))))
    elif type(dat) == list:#Instanciate from numbers
      self.triggers = dat[0]
      self.calor = dat[1]
      self.dsssd = dat[2]
      self.side = dat[3]
      self.single = dat[4]
      self.compton = dat[5]
      self.name = dat[6]
      if len(dat) > 8:
        self.theta = dat[7]
        self.phi = dat[8]
      else:
        self.theta = 180
        self.phi = 0
    elif dat is None:#Instanciate an empty Trafile (mainly for type check purposes)
      pass
    elif str(type(dat)) == str(type(Trafile(None))):#Instanciate from Trafile
      for e in dat.__dict__:
        setattr(self, e, getattr(dat, e))
    else:#datatype unknown
      raise TypeError("{} could not instanciate Trafile".format(dat))

  def __add__(self, other):
    """
    Implements built-in + operator for Trafile by adding each count-rate field in the two Trafiles
    :param other: Trafile
    :returns: Trafile
    """
    if str(type(other)) == str(type(self)):
      return Trafile([self.triggers+other.triggers, self.calor+other.calor, self.dsssd+other.dsssd, self.side+other.side, self.single+other.single, self.compton+other.compton, "{}+{}".format(self.name, other.name), self.theta, self.phi])
    else:
      raise TypeError("{} is not a Trafile")

  def __iadd__(self, other):
    """
    Implements incrementation (built-in +=) for Trafiles
    Keeps coordinates of the first Trafile
    :param other: Trafile
    """
    if str(type(other)) == str(type(self)):
      self.triggers += other.triggers
      self.calor += other. calor
      self.dsssd += other.dsssd
      self.side += other.side
      self.single += other.single
      self.compton += other.compton
      self.name = "{}+{}".format(self.name, other.name)
    else:
      raise TypeError("{} is not a Trafile")

  def __mul__(self, other):
    """
    Implements built-in * operator for Trafile by multiplicating each count-rate field by the other parameter
    """
    return Trafile([self.triggers*other, self.calor*other, self.dsssd*other, self.side*other, self.single*other, self.compton*other, self.name, self.theta, self.phi])

  def __rmul__(self, other):
    return self*other

  def setCompton(self, ergCut=(0, 460)):
    """
    Sets as number of Compton counts the number of events with CE 
    """
    if not(hasattr(self, "CE")):
      raise RuntimeError("Current Trafile has been instanciated without the required CE field for this method.")
    self.compton = len(np.asarray(inwindow(np.array(self.CE), ergCut)).nonzero()[0])

  def SNR_(self, item, background, C=0):
    """
    Calculates the SNR of this event
    :param item: str, member of Trafile to consider for calculation, ex="triggers"
    :param background: int or Trafile, background to consider. If Trafile, the same item will be considered.
    :param C: int, default=0
    :returns: float, SNR
    """
    if type(background) == float or type(background) == int:
      S, B = getattr(self, item), background
      #return SNR(getattr(self, item), background, C)
    elif str(type(background)) == str(type(Trafile(None))):
      if item == "CeBr3":
        S = self.calor + self.side
        B = background.calor + background.side
      else:
        S, B = getattr(self, item), getattr(background, item)
      #return SNR(getattr(self, item), getattr(background, item), C)
    return SNR(S, B, C)

#  def MDP99_(self, background, mu100):
#    """
#    Calculates the MDP at the 99% confidence level of this event
#    :param background: int or float or Trafile, background to consider.
#    :param mu100: float
#    :returns: float, MDP99%
#    """
#    if type(background) == float or type(background) == int:
#      b = background
#    elif str(type(background)) == str(type(Trafile(None))):
#      b = background.compton
#    s = self.compton
#    return 4.29*np.sqrt(s+b)/(mu100*s)



########## CLASS TRAFILELIST ##########

class Trafilelist(list):
  """
  """
  def __init__(self, prefix="", lsoptions="", lb=True):
    """
    Instanciates Trafilelist
    Typically (when used in Trafiletab) contains all simulations of a single GRB seen by a single satellite, ordered by sim number.
    :param prefix: str, prefix of files to read (used in an ls command). Empty str "" returns an empty Trafilelist. PATH ?? TBC
    :param lsoptions: str, options to use with ls, default=""
    :param lb: bool, whether or not to display a loading bar
    """
    if prefix == "":#Instanciate an empty Trafilelist
      list.__init__(self, [])
    elif str(type(prefix)) == str(type(Trafilelist(""))):#Instanciate from Trafilelist
      list.__init__(self, [Trafile(e) for e in prefix])
    elif type(prefix) == list:#Instanciate from list
      return list.__init__(self, prefix)
    else:#Instanciate from files
      if lsoptions != "":
        lsoptions = lsoptions.strip()+" "
      flist = subprocess.getoutput("ls {}{}".format(lsoptions, prefix)).split("\n")
      # Ligne supplementaire pour prendre en compte que ls ne renvoit pas rien quand il ne trouve pas de fichiers
      if flist[0].startswith("ls: cannot access") : flist = []
      if lb:
        flist = LoadingBar(flist, smtg="{} files with prefix {}".format(len(flist), prefix))
      if prefix.endswith("/"):
        list.__init__(self, [Trafile("{}{}".format(prefix, e), items=["CE"], optanalysis=[treatCE]) for e in flist])
      elif flist.tot < 1:
        print("No file found with prefix {}".format(prefix))
        list.__init__(self, [])
      else:
        list.__init__(self, [Trafile(e, items=["CE"], optanalysis=[treatCE]) for e in flist])

  def __getitem__(self, v):
    """
    Defines how items are accessed in Trafilelist
    :param v: index-like or slice, item(s) to access
    :returns: Trafile or Trafilelist
    """
    if isinstance(v, slice):
      return Trafilelist(list.__getitem__(self, v))
    else:
      return list.__getitem__(self, v)

  def permute(self, i, j):
    """
    Permutes element i and j
    """
    self[i], self[j] = self[j], self[i]

  def shuffle(self):
    """
    Shuffles the list of Trafile
    """
    for i in range(len(self)):
      self.permute(i, np.random.randint(len(self)))

  def setCompton(self, ergCut=(0, 460)):
    """
    """
    for e in self: e.setCompton(ergCut)

#  def setmu100(self, mu100):
#    """
#    Set the mu100 function if necessary for further use
#    :param mu100: couple ([list of theta, list of phi], list of mu100) or None
#    """
#    if type(mu100) == tuple and len(mu100) == 2:
#      self.mu100 = lambda theta, phi: griddata(*mu100, (theta, phi), method="cubic")
#    else:
#      self.mu100 = None

  def show(self, item=None, projection="aitoff"):
    """
    Shows coordinates of each GRB in the frame of the instrument
    :param item: (str, background, C) or str or None
      if None: simple scatter plot
      if str: item of Trafile
      if tuple: item of Trafile to use for SNR calculation (or "MDP" for MDP calculation) and colorbar, number or Trafile background to consider for calculation, optionnal SNR constant or mu100
      default=None
    :param projection: "aitoff" OR "hammer" OR "", type of projection on the sky, default="aitoff"
    """
    if item is None:
      skyScatterPlot([e.theta for e in self], [e.phi for e in self], projection=projection)
    elif type(item) == str:
      skyScatterPlot([e.theta for e in self], [e.phi for e in self], [getattr(e, item) for e in self], projection=projection)
    else:
      skyScatterPlot([e.theta for e in self], [e.phi for e in self], [e.SNR_(*item) for e in self], projection=projection)
    plt.grid(True)
    plt.show()

  def SNRcut(self, item, background, SNRmin, SNRmax, C=0):
    """
    Selects events in a certain range of SNR
    :param item: str, Trafile member to consider for SNR calculation
    :param background: float or Trafile, background for SNR calculation
    :param SNRmin: float, lower bound of SNR range
    :param SNRmax: float, upper bound of SNR range
    :param C: float, SNR calculation constant
    :returns: Trafilelist
    """
    ret = Trafilelist()
    for e in self:
      s = e.SNR_(item, background, C)
      if s > SNRmin and s < SNRmax:
        ret.append(e)
    return ret

  def SNR_(self, item, background, C=0):
    """
    Calculate SNR (signal to noise ratio) for all Trafiles in the Trafilelist
    :param item: str, member of Trafile to consider for calculation, ex="triggers"
    :param background: int or Trafile, background to consider. If Trafile, the same item will be considered.
    :param C: int, default=0
    :returns: float, SNR
    """
    return [e.SNR_(item, background, C) for e in self]

  def cSNR_(self, item, background, C=0):
    """
    Calculate SNR (signal to noise ratio) for all Trafiles in the Trafilelist representing a detection by a constellation of satellites with the same background
    :param item: str, member of Trafile to consider for calculation, ex="triggers"
    :param background: int or Trafile, background to consider. If Trafile, the same item will be considered.
    :param C: int, default=0
    :returns: list of floats, SNR
    """
    return [e.SNR_(item, background*(str(e.name).count("+")+1), C) for e in self]

  def hist(self, item, n=0, label="", show=True):
    """
    Shows a histogram of SNRs
    :param item: tuple (str, background) or (str, background, C), item of Trafiles to use for SNR calculation, background to consider, optionnal SNR constant
    :param n: int or list, number of bins or bins, determined automagically if n=0, default=0
    :param label: str, label of this histogram's curve, not shown if evaluate to False, default=""
    :param show: bool, wether or not to print the legend (if applicable), the axes titles and the plot, default=True
    """
    if type(n)==int and n==0: n = int(np.sqrt(len(self)))+1
    if "MDP" in item[0]:
      y, x = np.histogram([e.MDP99_(item[1]) for e in self], bins=n)
      xlabel = "MDP"
    else:
      y, x = np.histogram([e.SNR_(*item) for e in self], bins=n)
      xlabel = "SNR"
#    plt.hist([e.SNR_(*item) for e in self], bins=int(np.sqrt(len(self)))+1)
    if label:
      plt.step(x[1:], y, label=label)
    else:
      plt.step(x[1:], y)
    if show:
      plt.legend()
      plt.xlabel(xlabel)
      plt.ylabel("Number of events")
      plt.show()

  def detect(self, item, background, SNRmin, nsim=0, C=0):
    """
    Calculates the probability of detection of a single GRB
    :param item: str, member of Trafile to consider for calculation, ex="triggers"
    :param background: int or Trafile, background to consider. If Trafile, the same item will be considered.
    :param SNRmin: float, minimum SNR value to consider the detection
    :param nsim: int, number of simulations per event in *.par file, default=0. If provided >0, normalizes the probability to that number instead of the number of simulations actually ran (=GRB in field of view).
    :param C: int, default=0
    :returns: float, probability
    """
    if nsim <= 0:
      return np.mean(np.where(np.array([e.SNR_(item, background, C) for e in self])>SNRmin, 1, 0))
    else:
      return np.sum(np.where(np.array([e.SNR_(item, background, C) for e in self])>SNRmin, 1, 0))/nsim
  
  def cdetect(self, item, background, SNRmin, nsim=0, C=0):
    """
    Calculate detection probabilities for a single GRB detected by a constellation of satellites with the same background
    :param item: str, member of Trafile to consider for calculation, ex="triggers"
    :param background: int or Trafile, background to consider. If Trafile, the same item will be considered.
    :param SNRmin: float, minimum SNR value to consider the detection
    :param nsim: int, number of simulations per event in *.par file, default=0. If provided >0, normalizes the probability to that number instead of the number of simulations actually ran (=GRB in field of view).
    :param C: int, default=0
    :returns: float
    """
    if nsim <= 0:
      return np.mean(np.where(np.array(self.cSNR_(item, background, C))>SNRmin, 1, 0))
    else:
      return np.sum(np.where(np.array(self.cSNR_(item, background, C))>SNRmin, 1, 0))/nsim

  def cCompImage(self, nsim=0):
    """
    Calculate the probability of creating a compton image for a single GRB detected by a constellation of satellites with the same background
    :param nsim: int, number of simulations per event in *.par file, default=0. If provided >0, normalizes the probability to that number instead of the number of simulations actually ran (=GRB in field of view).
    :returns: float
    """
    if nsim <= 0:
      return np.mean(np.where(np.array([sim.compton for sim in self])>50, 1, 0))
    else:
      return np.sum(np.where(np.array([sim.compton for sim in self])>50, 1, 0))/nsim


########## CLASS TRAFILELIST ##########

class Trafiletab(list):

  def __init__(self, prefix, namelist=None, nsim=1, nsat=1):
    """
    """
    self.nsim = nsim
    self.nsat = nsat
    self.namelist = namelist
    if type(prefix) == list:
      list.__init__(self, prefix)
    elif str(type(prefix)) == str(type(Trafiletab([]))):
      self.nsat = prefix.nsat
      self.nsim = prefix.nsim
      self.namelist = prefix.namelist
      list.__init__(self, prefix)
    elif type(prefix) == str and namelist is not None:
      list.__init__(self, [[Trafilelist("{}_{}-sat{}*inc1*.tra".format(prefix, name, i), lb=True) for name in LoadingBar(namelist, smtg="{} GRBs from {}".format(len(namelist), prefix))] for i in range(nsat)])
      # Ancienne ligne, il faut prendre en compte que certaines données sont en polarisee et en non pola
      #list.__init__(self, [[Trafilelist("{}_{}-sat{}*.tra".format(prefix, name, i), lb=True) for name in LoadingBar(namelist, smtg="{} GRBs from {}".format(len(namelist), prefix))] for i in range(nsat)])
    elif type(prefix) == str and prefix.endswith(".h5"):
      hf = h5py.File(prefix, "r")
      self.nsim = hf.get('nsim')[()]
      self.nsat = hf.get('nsat')[()]
      self.namelist = [e.tobytes().decode('utf-8') for e in hf.get('namelist')[()]]
#      self.nsat = len(hf.keys())
      data = [[] for i in range(self.nsat)]
#      self.namelist = list(hf.get("satellite #0").keys())
      satellites = []
      for key in hf.keys():
        if key.startswith("satellite"):
          satellites.append(key)
      for sat, key in enumerate(satellites):
        gsat = hf.get(key)
#        for i, grb in enumerate(gsat.keys()):
          #for e in grb:
          #  tmp = list(e)
          #  tmp.insert(6, "")
        data[sat] = [Trafilelist([Trafile(list(e)) for e in gsat.get(grb)]) for i, grb in enumerate(gsat.keys())]
      hf.close()
      list.__init__(self, data)
    else:
      raise TypeError("")

  def setCompton(self, ergCut=(0, 460)):
    """
    """
    for i in range(self.nsat):
      for grb in range(len(self.namelist)):
        self[i][grb].setCompton(ergCut)

  def getsims(self, constellation=None):
    """
    Gets all simulations of a given GRB for a constellation
    :param constellation: list of indices of satellites, by order in *.par file, default=None in which case all are considered
    :returns: Trafiletab
    """
    if constellation is None:
      #constellation = range(len(d))  Ne fonctionne pas car d n'est pas défini
      constellation = range(len(self))
    data = [[[] for e in self.namelist]]
    for grb, name in enumerate(self.namelist):
      #print(name)
      #print("nombre de detec : ", len(self[0][grb])+len(self[1][grb])+len(self[2][grb]))
      for s in range(self.nsim):
        tmp = []
        for i in constellation:
          # k est l'index auquel on peut trouver la simulation s pour le satellite i
          k = min(s, len(self[i][grb])-1) #+ petit entre le nombre de simu et le nombre de simu détecté
          # Si k=-1 cela veut dire qu'il n'y a pas de détection pour le satellite
          if k >= 0: 
            num_simu = self[i][grb][k].name if type(self[i][grb][k].name) == int else int(self[i][grb][k].name.split("_")[1].split("-")[-1])
          else:
            num_simu = -1
          # Si un satellite n'a pas de détection et que les suivant en ont, il ne faudra pas ajouter de simu pour cette iteration de s et donc considerer qu'il n'y a pas de detection, la boucle suivante sert donc a rendre k =-1
          while k >= 0 and num_simu > s:
            k -= 1 #Find efficiently the right sim number
            if k >= 0:
              num_simu = self[i][grb][k].name if type(self[i][grb][k].name) == int else int(self[i][grb][k].name.split("_")[1].split("-")[-1])
            else:
              num_simu = -1
          #print("grb : ", grb, "      itesimu : ", s, "valeur de num_simu", num_simu,"     sat :", i, "     nsimdetec :", len(self[i][grb]), "     val k :", k)
          if num_simu == s: 
            #print("sauvegarde ", self[i][grb][k].name)
            tmp.append(self[i][grb][k])
        #print(type(tmp), type(np.sum(tmp)))
        #print("nombre de detections parmi les satellites pour une simu : ", len(tmp), "\n")
        if len(tmp) > 0: data[0][grb].append(np.sum(tmp)) #np.sum([self[i][grb] for i in constellation]))
      #print("nombre de simus pour ce GRB : ", len(data[0][grb]), "\n==========================\n")
      data[0][grb] = Trafilelist(data[0][grb])
    return Trafiletab(data, self.namelist, self.nsim)
    #return Trafiletab([data])
#    return Trafiletab([[Trafilelist() for i, e in self.namelist]])

  def getgrb(self, grb, constellation=None):
    """
    :param grb: int or str (index or name of GRB)
    """
    if type(grb)==str:
      grb = self.namelist.index(grb)
    return self.getsims(constellation)[0][grb]

  def getsim(self, grb, n, constellation=None):
    """
    Gets trafile containing summed observations throughout the constellation
    :returns: Trafile
    """
    return self.getgrb(grb, constellation)[n]

  def SNR(self, item, background, C=0):
    """
    Calculates SNR (signal-to-noise ratio) for all GRBs in the Trafiletab
    """
    return [[e.SNR_(item, background, C) for e in self[sat]] for sat in range(self.nsat)]

  def cSNR(self, constellation, item, background, C=0):
    """
    Calculates the SNR for constellation detection
    ex:>c=Trafiletab("path/to/files")
       >c.write("file.h5")
       >c=Trafiletab("file.h5")
       >c.cSNR(...)
    :param constellation: iterable containing the numbers of the satellites included in the constellation
      numbers are 0 for the first one of the *.par parameter file, 1 for the next, ...
    ...
    """
    data = self.getsims(constellation)
    return [e.cSNR_(item, background, C) for e in data[0]]

  def detect(self, item, background, SNRmin, nsim=0, C=0):
    """
    Calculate detection probabilities for all GRBs detected by a satellite
    :param item: str, member of Trafile to consider for calculation, ex="triggers"
    :param background: int or Trafile, background to consider. If Trafile, the same item will be considered.
    :param SNRmin: float, minimum SNR value to consider the detection
    :param nsim: int, number of simulations per event in *.par file, default=0. If provided >0, normalizes the probability to that number instead of the number of simulations actually ran (=GRB in field of view).
    :param C: int, default=0
    :returns: list of floats, probabilities
    """
    return [[e.detect(item, background, SNRmin, nsim, C) for e in self[sat]] for sat in range(self.nsat)]

  def cdetect(self, item, background, SNRmin, constellation=None, nsim=0, C=0):
    """
    Calculate detection probabilities for a all GRBs simulated for a constellation of satellites with the same background
    :param item: str, member of Trafile to consider for calculation, ex="triggers"
    :param background: int or Trafile, background to consider. If Trafile, the same item will be considered.
    :param SNRmin: float, minimum SNR value to consider the detection
    :param constellation: iterable containing the numbers of the satellites included in the constellation
      numbers are 0 for the first one of the *.par parameter file, 1 for the next, ...
    :param nsim: int, number of simulations per event in *.par file, default=0. If provided >0, normalizes the probability to that number instead of the number of simulations actually ran (=GRB in field of view).
    :param C: int, default=0
    :returns: list of floats, probabilities
    """
    data = self.getsims(constellation)
    return [e.cdetect(item, background, SNRmin, nsim, C) for e in data[0]]

  def cCompImage(self, constellation=None, nsim=0):
    """
    Calculate the probability of creating a compton image for all GRB detected by a constellation of satellites with the same background
    :param nsim: int, number of simulations per event in *.par file, default=0. If provided >0, normalizes the probability to that number instead of the number of simulations actually ran (=GRB in field of view).
    :returns: list of float, probabilities
    """
    data = self.getsims(constellation)
    return [e.cCompImage(nsim) for e in data[0]]

  def write(self, fname):
    """
    Writes the data to a file
    """
    hf = h5py.File(fname, "w")
    hf.create_dataset('nsim', data=self.nsim)
    hf.create_dataset('nsat', data=self.nsat)
    hf.create_dataset('namelist', data=[np.void(bytes(e, 'utf-8')) for e in self.namelist])
    for sat in range(self.nsat):
      gsat = hf.create_group("satellite #{}".format(sat))
      for i, name in enumerate(self.namelist):
        #for j, e in enumerate(self[sat][i]):
        #  print("{},{}:{}-{}-{}-{}-{}-{}".format(i, j, e.triggers, e.calor, e.dsssd, e.side, e.single, e.compton))
        temp = [[e.triggers, e.calor, e.dsssd, e.side, e.single, e.compton, int(e.name.split("_")[1].split("-")[-1]), e.theta, e.phi] for e in self[sat][i]]
        #gsat.create_dataset("{}".format(e.name), data=[e.triggers, e.calor, e.dsssd, e.side, e.single, e.compton, e.theta, e.phi])
        gsat.create_dataset("{}".format(name), data=temp)
    hf.close()

def GRBdetrate(model, flux, bins, data, plots=True, label=None):
  """
  Calculate GRB detection rate from a model
  :param model: GRB rate as a function of one float (flux)
  :param flux: list of GRB flux in the same order as data
  :param bins: bins used to generate the model (fitting histogram)
  :param data: data from Trafiletab.cdetect, in the same order as flux
  :returns: GRB detection rate per peak flux bin
  """
  sort = [[] for i in range(len(bins)-1)]
  for i in range(len(data)):#Loop over GRBs
    for j in range(1, len(bins)):#Search bin
      if flux[i] >= bins[j-1] and flux[i] < bins[j]:
        sort[j-1].append(data[i])
  x = .5*(bins[1:]+bins[:-1])#Same len as sort
  P = [np.mean(e) if len(e) > 0 else 0 for e in sort]
  if plots:
    plt.plot(x, P, label=label)
    plt.xscale('log')
    #plt.show()
  return np.array(P)*model(x)


## File management utilities

def treatCE(s):
  """
  """
  return float(s[0])+float(s[4])

def makeSimFolder(path):
  """
  """
  if path.endswith("/"): path = path[:-1]
  folder = path.split("/")[-1]
  path = "/".join(path.split("/")[:-1])
  os.system("mkdir {}/{}sim".format(path, folder))
  flist = subprocess.getoutput("ls {}/{}".format(path, folder)).split("\n")
  for f in flist:
    if f.endswith(".sim.gz"):
      os.system("mv {0}/{1}/{2} {0}/{1}sim/".format(path, folder, f))


def namelistFromPar(parfile, output_others=True):
  """
  Generates namelist used in TrafileTab from the *.par parameter file used in mamr
  """
  with open(parfile) as f:
    lines = f.read().split("\n")
  infos = {'type':'', 'sttype':'', 'file':'', 'instrument':''}
  nsat = 0
  for line in lines:
    if line.startswith("@type"): infos["type"]=line.split(" ")[1]
    elif line.startswith("@prefix"): prefix=line.split(" ")[1]
    elif line.startswith("@sttype"): infos["sttype"]=line.split(" ")[1:]
    elif line.startswith("@file"): infos["file"]=line.split(" ")[1]
    elif line.startswith("@instrument"): infos["instrument"]=line.split(" ")[1]
    elif line.startswith("@simulationsperevent"): nsim=int(line.split(" ")[1])
    elif line.startswith("@satellite"): nsat+=1
  if infos["type"]!='GRBCatalog' or infos["sttype"]=='' or infos["file"]=='' or infos["instrument"]=='':
    raise ValueError("Informantion is missing in *.par file : {}\n{}".format(parfile, infos))
  c = Catalog(infos["file"], infos["sttype"])
  if output_others: 
    return prefix, c.name, nsim, nsat
  else:
    return c

def parFile(parfile):
  """
  Parses available positions for GRBs and satellite positions from a *.par file
  :param parfile: name of *.par file
  :returns: dict with entries 'position' and 'satellites'
  """
  with open(parfile) as f:
    lines = f.read().split("\n")
  infos = {'position':'', 'satellites':[]}
  for line in lines:
    if line.startswith("@position"): infos['position'] = [np.deg2rad(float(e)) for e in line.split(" ")[1:]] #sky positions (rad)
    elif line.startswith("@satellite"):
      temp = [float(e) for e in line.split(" ")[1:]]
      if len(temp) == 3:#satellite pointing
        dat = [np.deg2rad(temp[0]), np.deg2rad(temp[1]), temp[2], horizonAngle(temp[2])]
      else:#satellite orbital parameters
        #inclination, omega, nu = map(np.deg2rad, temp[:3]) #rad
        #thetasat = np.arccos(np.sin(inclination)*np.cos(nu)) #rad
        #phisat = np.arctan2( (np.cos(nu)*np.cos(inclination)*np.cos(omega)-np.sin(nu)*np.sin(omega)) , (-np.sin(nu)*np.cos(omega)-np.cos(nu)*np.cos(inclination)*np.sin(omega)) ) #rad
        # Extracting inclination, ohm, omega, respectively the inclination, the right ascention of the ascending node and the argument of periapsis
        inclination, ohm, omega = map(np.deg2rad, temp[:3])
        thetasat = np.arccos(np.sin(inclination)*np.sin(omega)) #rad
        phisat = np.arctan2( (np.cos(omega) * np.sin(ohm) + np.sin(omega) * np.cos(inclination) *np.cos(ohm)) , (np.cos(omega) * np.cos(ohm) - np.sin(omega) * np.cos(inclination) * np.sin(ohm)) ) #rad
        dat = [thetasat, phisat, temp[3], horizonAngle(temp[3])]
      infos['satellites'].append(dat)
  return infos

def decra2tp(dec, ra, s, unit="deg"):
  """
  Converts dec,ra (declination, right ascension) world coordinates into satellite coordinate
  :param dec: declination (except it is 0 at north pole, 90° at equator and 180° at south pole)
  :param ra : Right ascension (0->360°)
  :param s: satellite from infos['satellites']
  :param unit: unit in which are given dec and ra, default="deg"
  :returns: theta_sat, phi_sat in rad
  """
  if unit=='deg':
    dec, ra = np.deg2rad(dec), np.deg2rad(ra)
  theta = np.arccos( np.product(np.sin(np.array([dec, ra, s[0], s[1]]))) + np.sin(dec)*np.cos(ra)*np.sin(s[0])*np.cos(s[1]) + np.cos(dec)*np.cos(s[0]) )
  source = [np.sin(dec)*np.cos(ra), np.sin(dec)*np.sin(ra), np.cos(dec)]
  yprime = [-np.cos(s[0])*np.cos(s[1]), -np.cos(s[0])*np.sin(s[1]), np.sin(s[0])]
  xprime = [-np.sin(s[1]), np.cos(s[1]), 0]
  phi = np.mod(np.arctan2(np.dot(source, yprime), np.dot(source, xprime)), 2*np.pi)
  return theta, phi

def decrasat2world(dec, ra, s, unit="deg"):
  """
  Converts dec,ra (declination, right ascension) satellite coordinates into world coordinate
  :param dec: declination (except it is 0 at instrument zenith and 90° at equator)
  :param ra : Right ascension (0->360°)
  :param s: satellite from infos['satellites']
  :param unit: unit in which are given dec and ra, default="deg"
  :returns: theta_world, phi_world in rad
  """
  if unit=='deg':
    dec, ra = np.deg2rad(dec), np.deg2rad(ra)
  xworld = [-np.sin(s[1]), -np.cos(s[0])*np.cos(s[1]), np.sin(s[0])*np.cos(s[1])]
  yworld = [np.cos(s[1]), -np.cos(s[0])*np.sin(s[1]), np.sin(s[0])*np.sin(s[1])]
  zworld = [0, np.sin(s[0]), np.cos(s[0])]
  source = [np.sin(dec)*np.cos(ra), np.sin(dec)*np.sin(ra), np.cos(dec)]
  theta = np.arccos(np.dot(source, zworld))
  phi = np.mod(np.arctan2(np.dot(source, yworld), np.dot(source, xworld)), 2*np.pi)
  return theta, phi


def writeDat(fname, t, x, y):
  """
  :param fname: file name
  :param t: type, ex: "LOGLOG"
  :param x: first column
  :param y: second column
  """
  with open(fname, "w") as f:
    f.write("IP {}\n\n".format(t))
    for e, d in zip(x, y):
      f.write("DP {} {}\n".format(e, d))
    f.write("\nEN\n\n")


