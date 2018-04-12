#!/usr/bin/python
import sys
import numpy as np

def FreeEnergy(dosFile, T):
    E, g = np.loadtxt(dosFile, unpack = True)
    F = np.zeros([len(T)])
    emax = np.max(E)
    for i, beta in enumerate(1./T):
        F[i] = -1./beta*np.log(np.sum(np.exp(-beta*(E+emax))*g)) + emax
    return F

try:
    dos1fname = sys.argv[1]
    dos2fname = sys.argv[2]
    dh = float(sys.argv[3])
    L = int(sys.argv[4])
except IndexError:
    print """
    argv[1] : dos 1
    argv[2] : dos 2
    argv[3] : delta h(Zeeman field)
    argv[4] : L(system size)
"""
    exit(1)

# temperature grid
T = np.linspace(0.5, 4, 100)

F1 = FreeEnergy(dos1fname, T); F2 = FreeEnergy(dos2fname, T)

# m = dF/dh (m: magnetization, F: free energy, h: Zeeman field)
m = (F1 - F2)/(dh*L**2)

np.savetxt('L%d-h%s.out'%(L, str(dh)), np.array([T, m]).T)
