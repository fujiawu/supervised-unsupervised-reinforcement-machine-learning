
def mushroom_dict():
   # dictionaries for parsing inputs
   dicts = []

   dict = {"e":float(1), "p": float(0)} #label
   dicts.append(dict)

   dict = {"b":float(0), "c":float(1), "x":float(2), #1
        "f":float(3), "k":float(4), "s":float(5)}
   dicts.append(dict)

   dict = {"f":float(0), "g":float(1), "y":float(2),"s":float(3)} #2
   dicts.append(dict)

   dict = {"n":float(0), "b":float(1), "c":float(2), "g":float(3), #3
        "r":float(4), "p":float(5), "u":float(6), "e":float(7),
        "w":float(8), "y":float(9)}
   dicts.append(dict)

   dict = {"t":float(0), "f":float(1)}  #4
   dicts.append(dict)

   dict = {"a":float(0), "l":float(1), "c":float(2), "y":float(3), #5
        "f":float(4), "m":float(5), "n":float(6), "p":float(7),
        "s":float(8)}
   dicts.append(dict)

   dict = {"a":float(0), "d":float(1), "f":float(2), "n":float(3)} #6
   dicts.append(dict)

   dict = {"c":float(0), "w":float(1), "d":float(2), "y":float(3)} #7
   dicts.append(dict)

   dict = {"b":float(0), "n":float(1)} #8
   dicts.append(dict)

   dict = {"k":float(0), "n":float(1), "b":float(2), "h":float(3), #9
        "g":float(4), "r":float(5), "o":float(6), "p":float(7),
        "u":float(8), "e":float(9), "w":float(10), "y":float(11)}
   dicts.append(dict)

   dict = {"e":float(0), "t":float(1)} #10
   dicts.append(dict)

   dict = {"b":float(0), "c":float(1), "u":float(2), "e":float(3),
        "z":float(4), "r":float(5), "?":float(6)} #11
   dicts.append(dict)

   dict = {"f":float(0), "y":float(1), "k":float(2), "s":float(3)} #12
   dicts.append(dict)

   dict = {"f":float(0), "y":float(1), "k":float(2), "s":float(3)} #13
   dicts.append(dict)

   dict = {"n":float(0), "b":float(1), "c":float(2), "g":float(3), #14
        "o":float(4), "p":float(5), "e":float(6), "w":float(7),
        "y":float(8)}
   dicts.append(dict)

   dict = {"n":float(0), "b":float(1), "c":float(2), "g":float(3), #15
        "o":float(4), "p":float(5), "e":float(6), "w":float(7),
        "y":float(8)}
   dicts.append(dict)

   dict = {"p":float(0), "u":float(1)} #16
   dicts.append(dict)

   dict = {"n":float(0), "o":float(1), "w":float(2), "y":float(3)} #17
   dicts.append(dict)

   dict = {"n":float(0), "o":float(1), "t":float(2)} #18
   dicts.append(dict)

   dict = {"c":float(0), "e":float(1), "f":float(2), "l":float(3), #19
        "n":float(4), "p":float(5), "s":float(6), "z":float(7)}
   dicts.append(dict)

   dict = {"k":float(0), "n":float(1), "b":float(2), "h":float(3), #20
        "r":float(4), "o":float(5), "u":float(6), "w":float(7), 
        "y":float(8)}
   dicts.append(dict)

   dict = {"a":float(0), "c":float(1), "n":float(2), "s":float(3), #21
        "v":float(4), "y":float(5)}
   dicts.append(dict)

   dict = {"g":float(0), "l":float(1), "m":float(2), "p":float(3), #22
        "u":float(4), "w":float(5), "d":float(6)}
   dicts.append(dict)

   return dicts

def car_dict():
   dicts = []
   dict = {"vhigh":float(1), "high":float(2)/3, "med":float(1)/3, "low": float(0)}
   dicts.append(dict)
   dicts.append(dict)
   dict = {"5more":float(1), "4":float(2)/3, "3":float(1)/3, "2":float(0)}
   dicts.append(dict)
   dict = {"more":float(1), "4":float(0.5), "2":float(0)}
   dicts.append(dict)
   dict = {"big":float(1), "med":float(0.5), "small":float(0)}
   dicts.append(dict)
   dict = {"high":float(1), "med":float(0.5), "low":float(0)}
   dicts.append(dict)
   #dict = {"vgood":[1,0,0,0], "good":[0,1,0,0], "acc":[0,0,1,0], "unacc":[0,0,0,1]}
   dict = {"vgood":0, "good":1, "acc":2, "unacc":3}
   dicts.append(dict)
   return dicts

# helper function for input parsing
def parse2double(line,dicts):
    line = line.strip().split(',')
    data = []
    for i, item in enumerate(line):
       data.append(dicts[i][item])
    return data

# a data set class
class LoadData:
    def __init__(self, name):
       if "car" in name:
          self.load_data("car.data", car_dict())
       elif "mushroom" in name:
          self.load_data("mushroom.data", mushroom_dict())
       else:
          self.data=[]
          self.lables=[]
        
    def load_data(self, filename, dicts):
       f = open(filename, 'r')
       raw = f.readlines()
       #shuffle(raw)
       self.data = [[] for line in raw]
       self.labels = [[] for line in raw]
       for i,line in enumerate(raw):
          item = parse2double(line,dicts)
          self.data[i] = item[1:]
          self.labels[i] = item[0]


