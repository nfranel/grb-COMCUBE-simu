import numpy as np

# Version 2, Created by Adrien Laviron, updated by Nathan Franel

class Catalog:

  def __init__(self, data=None, sttype=None):
    """
    Instanciates a catalog
    :param data: None or string, data to put in the catalog
    :param sttype: See Catalog.fill
    """
    self.name = None
    self.t90 = None
    self.length = 0
    self.data = None
    self.sttype = None
    self.dec = None
    self.ra = None
    if not (data is None or sttype is None):
      self.fill(data, sttype)

  def __len__(self):
    """
    Makes use of built-in len function
    """
    return self.length

  def formatsttype(self):
    """
    Formats self.sttype, the standardized type of text data file
    """
    for i in range(5):
      if self.sttype[i] == "n":
        self.sttype[i] = "\n"
      if self.sttype[i] == "t":
        self.sttype[i] = "\t"
      if i % 2 == 0:
        if type(self.sttype[i]) == str and self.sttype[i].startswith('['):
          self.sttype[i] = self.sttype[i][1:-1].split(',')
        else:
          self.sttype[i] = int(self.sttype[i])

  def tofloat(self, item, default=0):
    """
    Convert an item of all events to float
    :param item: str, item
    :param default: default value, default=0
    """
    if not (hasattr(self, item)):
      raise AttributeError("Catalog does not contain item {}".format(item))
    for i in range(self.length):
      try:
        getattr(self, item)[i] = float(getattr(self, item)[i])
      except ValueError:
        getattr(self, item)[i] = default

  def tofloats(self, items, defaults=0):
    """
    Convert several items of all events to float
    :param items: list of str, items
    :param defaults: value or list of values, default values, default=0
    """
    if not (hasattr(defaults, "__iter__")):
      defaults = np.full(len(items), defaults)
    for item, default in zip(items, defaults):
      self.tofloat(item, default)

  def fill(self, data, sttype):
    """
    Fills a Catalog with data
    :param data: string, data file name
    :param sttype: iterable of len 5: first header event (int), event separator (str), first event (int), item separator (str), last event (int) OR list of the sources wanted (list)
    """
    self.data = data
    self.sttype = sttype
    self.formatsttype()
    with open(data) as f:
      d = f.read().split(sttype[1])
    if type(sttype[4]) == int:
      events = d[sttype[2]:sttype[4]]
      if events[-1] == '':
        events = events[:-1]
    elif type(sttype[4]) == list:
      events = d[sttype[2]:]
      if events[-1] == '':
        events = events[:-1]
      events = [event for event in events if event.split(sttype[3])[1] in sttype[4]]
    else:
      events = []
    self.length = len(events)
    if events[-1] == "":
      self.length -= 1
    header = d[sttype[0]]
    items = [i.strip() for i in header.split(sttype[3])]
    c = 0  # Compteur d'Empty
    for i in range(len(items)):
      if items[i] == "":
        items[i] = "Empty{}".format(c)
        c += 1
    for item in items:
      setattr(self, item, list())
    for e in events:
      for item, value in zip(items, e.split(sttype[3])):
        getattr(self, item).append(value)

  def items(self):
    """
    List all knowns items
    """
    return list(self.__dict__.keys())[3:]
