# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
from scipy.optimize import curve_fit
from inspect import signature
# Developped modules imports
from funcmod import *


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

  def disp(self):
    """
    Prints statistics about the polarization analysis
    """
    if self.comment == "modulation":
      print("\nPolarization analysis:")
      pa = (self.popt[0] + (90 if self.popt[1] < 0 else 0)) % 180
      print("\tModulation        :  {}+-{}".format(abs(self.popt[1]), np.sqrt(self.pcov[1][1])))
      print("\tPolarization angle: ({}+-{}) deg".format(pa, np.sqrt(self.pcov[0][0])))
      print("\tSource flux       :  {}+-{}".format(self.popt[2], np.sqrt(self.pcov[2][2])))
      print("\tFit goodness      : {}\n".format(self.q2 / (len(self.x) - self.nparam)))
    elif self.comment == "constant":
      print("\nConstant fit:")
      print("\tFit goodness      : {}\n".format(self.q2 / (len(self.x) - self.nparam)))
    else:
      print("\n{}: Unknown fit type - displaying raw results".format(self.comment))
      print(self.popt)
      print(np.sqrt(np.diag(self.pcov)))
      print(self.pcov)
      print("Q^2 / ndof: {}\n".format(self.q2 / (len(self.x) - self.nparam)))
