import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

with open('outputData.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    
tick = []
bodies = []
mass =[]
kinetic = []
potential = []
total = []
for line in data:
    if (len(line) >= 6):
        tick.append(int(float(line[0])))
        bodies.append(int(float(line[1])))
        mass.append(float(line[2]))

#print(data)

plt.plot(tick, bodies, label='Number of bodies')
plot.plot(tick, )

plt.legend()
plt.show()
