import numpy as np
import matplotlib.pyplot as plt


class Finite_Element_Solver():

    def Barycenter_Coords(self,xi,xj,xk,x,y):
        #Defining barycenter coordinates in terms of the vertices of the triangle and the physical cylindrical coordinates
        a = (xk[0]*(xi[1]-xj[1])+ xi[0]*(xj[1]-xk[1])+ xj[0]*(xk[1]-xi[1]))**(-1)
        l1 = a*(xj[0]*xk[1]-xj[1]*xk[0] +R*xj[1]-Z*xj[0]+ Z*xk[0]-R*xk[1])
        l2 = a*(xk[0]*xi[1]-xk[1]*xi[0] +R*xk[1]-Z*xk[0]+ Z*xi[0]-R*xi[1])
        l3 = a*(xi[0]*xj[1]-xi[1]*xj[0] +R*xi[1]-Z*xi[0]+ Z*xj[0]-R*xj[1])
        J = a*np.array([xk[1]-xi[1],xi[1]-xj[1]],[xi[0]-xk[0],xj[0]-xi[0]])
        
        return l1, l2, l3, J, a
    def Basis_Functions(self,xi,xj,xk,x,y):
        #Using cubic basis functions for each triangle and written in terms of Barycenter coordinates
        l1, l2, l3 = self.Barycenter_Coords(xi,xj,xk,x,y)
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
    def IntegrateTriangle(self, f):
        #Using 16-point Gauss quadrature rule to perform an integral over the fundamental triangle in barycenter coordinates

        #Defines weights and points over which we will sum
        pts = np.array([(1/3, 1/3, 1/3), 
                                (0.081414823414554, 0.459292588292723, 0.459292588292723),
                                (0.459292588292723, 0.081414823414554, 0.459292588292723),
                                (0.459292588292723, 0.459292588292723, 0.081414823414554),
                                (0.658861384496480, 0.170569307751760, 0.170569307751760),
                                (0.170569307751760, 0.658861384496480, 0.170569307751760),
                                (0.170569307751760, 0.170569307751760, 0.658861384496480),
                                (0.898905543365938, 0.050547228317031, 0.050547228317031),
                                (0.050547228317031, 0.898905543365938, 0.050547228317031),
                                (0.050547228317031, 0.050547228317031, 0.898905543365938),
                                (0.008394777409958, 0.263112829634638, 0.728492392955404),
                                (0.008394777409958, 0.728492392955404, 0.263112829634638),
                                (0.263112829634638, 0.008394777409958, 0.728492392955404),
                                (0.263112829634638, 0.728492392955404, 0.008394777409958),
                                (0.728492392955404, 0.008394777409958, 0.263112829634638),
                                (0.728492392955404, 0.263112829634638, 0.008394777409958)])
        weights = np.array([0.144315607677787, 
                   0.095091634267285, 0.095091634267285, 0.095091634267285,
                   0.103217370534718, 0.103217370534718, 0.103217370534718,
                   0.032458497623198, 0.032458497623198, 0.032458497623198,
                   0.027230314174435, 0.027230314174435, 0.027230314174435,
                   0.027230314174435, 0.027230314174435, 0.027230314174435])
        I = 0
        for (l1,l2,l3),w in zip(pts,weights):
            I += w*f(l1,l2,l3)
        return 0.5*I

    def Square_Mesh(Nx, Ny, x_i, x_f, y_i, y_f):
        #Constructs a square mesh for FEM
        x = np.linspace(x_i, x_f, Nx)
        y = np.linspace(y_i, y_f, Ny)

        T = np.zeros((3,2,2*(Nx-1)*(Ny-1)))
        for n in range(Ny-1):
            for m in range(Nx-1):
                T[0,:,2*(m+(Nx-1)*n)] = np.array([x[m],y[n]]) 
                T[1,:,2*(m+(Nx-1)*n)] = np.array([x[m],y[n+1]])
                T[2,:,2*(m+(Nx-1)*n)] = np.array([x[m+1],y[n]]) 
                
                T[0,:,2*(m+(Nx-1)*n)+1] = np.array([x[m+1],y[n]]) 
                T[1,:,2*(m+(Nx-1)*n)+1] = np.array([x[m],y[n+1]])
                T[2,:,2*(m+(Nx-1)*n)+1] = np.array([x[m+1],y[n+1]]) 
        return T
    def StiffnessMatrix(self,Nx, Ny, x_i, x_f, y_i, y_f):    
        #Construct stiffness matrix in FEM
        T = self.Square_Mesh(Nx, Ny, x_i, x_f, y_i, y_f)
        #Compute local contribution to stiffness matrix, which will be a 10x10 symmetric matrix
        #Add all local contributions together to compute 
        
