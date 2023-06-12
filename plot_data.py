import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

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
        kinetic.append(float(line[3]))
        potential.append(float(line[4]))
        total.append(abs(float(line[5])))

#print(data)

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.semilogy(tick, kinetic, label='Kinetic Energy')
ax.semilogy(tick, potential, label='Potential Energy')
ax.semilogy(tick, total, label='Total Energy')
ax.set_title('System Energy')
ax.set_xlabel('Tick')
ax.set_ylabel('Energy (J)')
ax.grid(which='both', axis='both')
ax.legend()

ax = fig.add_subplot(2, 1, 2)
ax.plot(tick, bodies)
ax.set_title('Body Count')
ax.set_xlabel('Tick')
ax.set_ylabel('Bodies')
ax.grid(which='both', axis='both')

'''
plt.plot(tick, bodies, label='Number of bodies')
plt.plot(tick, kinetic, label='Kinetic Energy')
plt.plot(tick, potential, label='Potential Energy')
plt.plot(tick, total, label='Net Energy')
'''
ax.legend()
plt.show()
