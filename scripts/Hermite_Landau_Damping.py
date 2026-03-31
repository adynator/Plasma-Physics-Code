import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import math as math

class Hermite_Landau:
    #Class to solve a (1+1)d linearised kinetic equation with an electrostatic potential 
    #and a Lenard–Bernstein collision operator. The problem was expanded in Hermite polynomials
    #in the velocity direction and Fourier modes in the position direction. 
    
    def __init__(self, alpha, k, v):
        #Alpha is the constant that goes into the definition of the elctrostatic potential 
        #and depending on the interpretation captures different physics.
        #k is the wavenumber.
        #v is the collision frequency
        self.alpha = alpha
        self.k = k
        self.v = v
        

    def initial_condition(self, g0):
        self.g0 = g0

    def solve(self, N, M, dt):

        #Building the tri-diagonal matrix used in the problem
        a = np.zeros(N-1, dtype = complex)
        b = np.zeros(N, dtype = complex)
        c = np.zeros(N-1, dtype = complex)
        d = self.g0.astype(complex)

        a[0] = 1j*self.k*dt*(1+self.alpha)/math.sqrt(2)
        b[0] = 1
        b[1] = 1
        c[0] = 1j*self.k*dt/math.sqrt(2)
        
        for i in range(1,N):
            if i != N-1 and i != 1:
                a[i] = 1j*self.k*dt*math.sqrt(i+1)/math.sqrt(2)
                b[i] = 1+ (i**4)*dt*self.v
                c[i] = 1j*self.k*dt*math.sqrt(i+1)/math.sqrt(2)
            elif i == 1:
                a[i] = 1j*self.k*dt*math.sqrt(i+1)/math.sqrt(2)
                b[i] = 1
                c[i] = 1j*self.k*dt*math.sqrt(i+1)/math.sqrt(2)
            else: 
                 b[i] = 1+ (i**4)*dt*self.v

        
        #Solves for the g_{m,k} as a function of time using the Tridiagonal function
        g = np.zeros((self.g0.size,M), dtype = complex)
        t = np.zeros(M)
        g[:,0] = d
        for i in range(M-1):
            g[:,i+1] = self.TriDiagonal(a,b,c,g[:,i])
            t[i+1] = t[i]+ dt

        #Builds the electrostatic potential and free energy as functions of time
        phi = (self.alpha**2)*(np.abs(g[0,:]))**2
        W =  np.zeros(M)
        for i in range(M):
            W[i] = 0.5*self.alpha*np.abs(g[0,i])**2 + 0.5*np.sum(np.abs(g[:,i])**2)
        
        return phi, W, g, t

    def TriDiagonal(self,a,b,c,d):
        #Solves Mx = d where M is an nxn tri-diagonal matrix using the Thomas algorithm
        #a and c are (n-1)-dim vectors and correspond to the two off-diagonals
        #b and d are n-dim vectors with b being the diagonal entries of M
        n = b.size
        B = np.zeros(n, dtype = complex)
        C = np.zeros(n-1, dtype = complex)
        D = np.zeros(n, dtype = complex)
        x = np.zeros(n, dtype = complex)
        B[0] = b[0]
        C[0] = c[0]
        D[0] = d[0]
        for i in range(n-1):
            if i != n-2 :
                B[i+1] = B[i]*b[i+1] - C[i]*a[i]
                C[i+1] = B[i]*c[i+1]
                D[i+1] = B[i]*d[i+1] - D[i]*a[i]
            else:
                B[i+1] = B[i]*b[i+1] - C[i]*a[i]
                D[i+1] = B[i]*d[i+1] - D[i]*a[i]
        x[n-1] = D[n-1]/B[n-1]
        for i in range(n-1):
            x[n-2-i] = (D[n-2-i] - C[n-2-i]*x[n-1-i])/B[n-2-i]
        return x

problem = Hermite_Landau(1,2*np.pi, 10**(-2))
N = 128
g0 = np.zeros(N)
g0[0] = 1
problem.initial_condition(g0)
phi, W, g, t = problem.solve(N, 40000, 0.0001)
