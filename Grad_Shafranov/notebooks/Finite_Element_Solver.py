import numpy as np
import matplotlib.pyplot as plt


class Finite_Element_Solver():

    def Barycenter_Coords(self,xi,xj,xk,R,Z):
        #Defining barycenter coordinates in terms of the vertices of the triangle and the physical cylindrical coordinates
        a = (xk[0]*(xi[1]-xj[1])+ xi[0]*(xj[1]-xk[1])+ xj[0]*(xk[1]-xi[1]))**(-1)
        l1 = a*(xj[0]*xk[1]-xj[1]*xk[0] +R*xj[1]-Z*xj[0]+ Z*xk[0]-R*xk[1])
        l2 = a*(xk[0]*xi[1]-xk[1]*xi[0] +R*xk[1]-Z*xk[0]+ Z*xi[0]-R*xi[1])
        l3 = a*(xi[0]*xj[1]-xi[1]*xj[0] +R*xi[1]-Z*xi[0]+ Z*xj[0]-R*xj[1])
        
        return l1, l2, l3
    def Basis_Functions(self,xi,xj,xk,R,Z):
        #Using cubic basis functions for each triangle and written in terms of Barycenter coordinates
        l1, l2, l3 = self.Barycenter_Coords(xi,xj,xk,R,Z)
        phi = np.zeros(9)
        phi[0] = 0.5*l1*(3*l1-1)*(3*la1-2)
        phi[1] = 0.5*l2*(3*l2-1)*(3*l2-2)
        phi[2] = 0.5*l3*(3*l3-1)*(3*l3-2)
        phi[3] = 4.5*l1*l2*(3*l1-1)
        phi[4] = 4.5*l1*l2*(3*l2-1)
        phi[5] = 4.5*l2*l3*(3*l2-1)
        phi[6] = 4.5*l2*l3*(3*l3-1)
        phi[7] = 4.5*l1*l3*(3*l3-1)
        phi[8] = 4.5*l1*l3*(3*l1-1)
        phi[9] = 27*l1*l2*l3

        #Gradients of basis functions in barycenter coordinates
        grad = np.zeros((9,2))
        g1 = 0.5 * (27*l1*l1 - 18*l1 + 2)
        g2 = 0.5 * (27*l2*l2 - 18*l2 + 2)
        g3 = 0.5 * (27*l3*l3 - 18*l3 + 2)

        grad[0] = np.array([-g1, -g1])
        grad[1] = np.array([0, g2])
        grad[2] = np.array([g3, 0])
        grad[3] = np.array([4.5 * (l2 * (6*l1 - 1) * (-1.0) + l1 * (3*l1 - 1) * 0.0), 4.5 * (l2 * (6*l1 - 1) * (-1.0) + l1 * (3*l1 - 1) * 1.0)])
        grad[4] = np.array([4.5 * (l2 * (3*l2 - 1) * (-1.0) + l1 * (6*l2 - 1) * 0.0), 4.5 * (l2 * (3*l2 - 1) * (-1.0) + l1 * (6*l2 - 1) * 1.0)])
        grad[5] = np.array([4.5 * (l3 * (6*l2 - 1) * 0.0 + l2 * (3*l2 - 1) * 1.0), 4.5 * (l3 * (6*l2 - 1) * 1.0 + l2 * (3*l2 - 1) * 0.0)])
        grad[6] = np.array([4.5 * (l3 * (3*l3 - 1) * 0.0 + l2 * (6*l3 - 1) * 1.0), 4.5 * (l3 * (3*l3 - 1) * 1.0 + l2 * (6*l3 - 1) * 0.0)])
        grad[7] = np.array([4.5 * (l1 * (6*l3 - 1) * 1.0 + l3 * (3*l3 - 1) * (-1.0)), 4.5 * (l1 * (6*l3 - 1) * 0.0 + l3 * (3*l3 - 1) * (-1.0))])
        grad[8] = np.array([4.5 * (l1 * (3*l1 - 1) * 1.0 + l3 * (6*l1 - 1) * (-1.0)), 4.5 * (l1 * (3*l1 - 1) * 0.0 + l3 * (6*l1 - 1) * (-1.0))])
        grad[9] = np.array([27.0 * (-l2*l3 + -l1*l3 + l1*l2 * 0.0), 27.0 * (-l2*l3 + l1*(-l3) + l1*l2 * 1.0)])

        return phi, grad
    def IntegrateTriangle():
        #Using 7-point Gauss quadrature rule to perform integral over fundamental triangle in barycenter coordinates
    
    def StiffnessMatrix():    
        #Construct stiffness matrix in FEM
