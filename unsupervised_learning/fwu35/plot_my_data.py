import matplotlib.pyplot as plt
import sys
import numpy as np

raw = sys.stdin.readlines()
n = [[] for line in raw]
time = [[] for line in raw]
homo = [[] for line in raw]
si = [[] for line in raw]
for i,line in enumerate(raw):
    line = line.strip().split()
    n[i] = int(line[0])
    time[i] = float(line[1][0:5])
    homo[i] = float(line[2])
    si[i] = float(line[7])

f, ax1 = plt.subplots()
h, = ax1.plot(n, homo, 'bo-', label='homo score')
s, = ax1.plot(n, si, 'gs-', label='silhouette score')
ax1.set_xlabel("Number of Clusters", fontsize=14)
ax1.set_ylabel("Score", fontsize=14)
ax1.set_ylim([0,0.3])
#for tl in ax1.get_yticklabels():
#    tl.set_color('b')

ax2 = ax1.twinx()
t, = ax2.plot(n, time, 'r-', label='time(s)')
ax2.set_ylim([0,14])
ax2.set_ylabel('time(s)', fontsize=14)
#for tl in ax2.get_yticklabels():
#    tl.set_color('r')

#plt.title("KMeans on car.data",fontsize=20)
#plt.title("KMeans on mushroom.data",fontsize=20)
plt.title("EM on car.data",fontsize=20)
#plt.title("EM on mushroom.data",fontsize=20)
plt.legend(handles=[t,h,s], loc=2)
plt.show()
