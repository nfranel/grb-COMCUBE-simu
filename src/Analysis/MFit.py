# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from inspect import signature

# Developped modules imports


class Fit:
  """
  Fit container
  :field f:       function, function fitted to data
  :field x:       np.array, data parameter
  :field y:       np.array, data
  :field popt:    np.array, optimum parameters
  :field pcov:    np.array, covariance matrix
  :field comment: str,      comment on the fit (ex: type, human readable name of function, ...)
  :field q2:      float,    Q^2 value of the fit
  :field nparam:  int,      number of parameters of the function
  """

  def __init__(self, f, x, y, yerr=None, bounds=None, comment=""):
    """
    Instanciates a Fit
    :param f:       function, function fitted to data
    :param x:       np.array, data parameter
    :param y:       np.array, data
    :param comment: str,      comment on the fit (ex: type, human readable name of function, ...)
    :returns:       Correctly instanciated Fit
    """
    self.f = f
    self.x = x
    self.y = y
    if bounds is None:
      self.popt, self.pcov = curve_fit(f, x, y, sigma=yerr)[:2]
    else:
      self.popt, self.pcov = curve_fit(f, x, y, sigma=yerr, bounds=bounds)[:2]
    self.comment = comment
    yf = f(x, *self.popt)
    self.q2 = np.sum((y - yf) ** 2)
    self.nparam = len(signature(f).parameters) - 1
    self.p_value = chi2.sf(self.q2, len(self.x) - self.nparam)

  def disp(self):
    """
    Prints statistics about the polarization analysis
    """
    if self.comment == "modulation":
      print("\nPolarization analysis:")
      pa = (self.popt[0] + (90 if self.popt[1] < 0 else 0)) % 180
      print(f"\tModulation         :  {abs(self.popt[1])}+-{np.sqrt(self.pcov[1][1])}")
      print(f"\tPolarization angle : ({pa}+-{np.sqrt(self.pcov[0][0])}) deg")
      print(f"\tSource flux        :  {self.popt[2]}+-{np.sqrt(self.pcov[2][2])}")
      print(f"\tFit goodness       : {self.p_value}\n")
    elif self.comment == "constant":
      print(f"\nConstant fit:")
      print(f"\tFit goodness       : {self.p_value}\n")
    else:
      print(f"\n{self.comment}: Unknown fit type - displaying raw results")
      print(self.popt)
      print(np.sqrt(np.diag(self.pcov)))
      print(self.pcov)
      print(f"Q^2 / ndof: {self.p_value}\n")
