import numpy as np
import matplotlib.pyplot as plt


class Finite_Element_Solver():

    def Barycenter_Coords(self,xi,xj,xk):
        #Defining barycenter coordinates in terms of the vertices of the triangle and the physical cylindrical coordinates
        a = (xk[0]*(xi[1]-xj[1])+ xi[0]*(xj[1]-xk[1])+ xj[0]*(xk[1]-xi[1]))
        def l1(x,y):
            return (xj[0]*xk[1]-xj[1]*xk[0] +x*xj[1]-y*xj[0]+ y*xk[0]-x*xk[1])/a
        def l2(x,y):
            return (xk[0]*xi[1]-xk[1]*xi[0] +x*xk[1]-y*xk[0]+ y*xi[0]-x*xi[1])/a
        def l3(x,y):
            return (xi[0]*xj[1]-xi[1]*xj[0] +x*xi[1]-y*xi[0]+ y*xj[0]-x*xj[1])/a
            
        J = np.array([[xk[1]-xi[1],xi[1]-xj[1]],[xi[0]-xk[0],xj[0]-xi[0]]])/a
        
        return l1, l2, l3, J, a
    def Basis_Functions(self,xi,xj,xk):
        #Using cubic basis functions for each triangle and written in terms of Barycenter coordinates
        def phi1(l1,l2,l3):
            return 0.5*l1*(3*l1-1)*(3*l1-2)
        def phi2(l1,l2,l3):
            return 0.5*l2*(3*l2-1)*(3*l2-2)
        def phi3(l1,l2,l3):
            return 0.5*l3*(3*l3-1)*(3*l3-2)
            
        def phi4(l1,l2,l3):
            return 4.5*l1*l2*(3*l1-1)
        def phi5(l1,l2,l3):
            return 4.5*l1*l2*(3*l2-1)
        def phi6(l1,l2,l3):
            return 4.5*l2*l3*(3*l2-1)
        def phi7(l1,l2,l3):
            return 4.5*l2*l3*(3*l3-1)
        def phi8(l1,l2,l3):
            return 4.5*l1*l3*(3*l3-1)
        def phi9(l1,l2,l3):
            return 4.5*l1*l3*(3*l1-1)
            
        def phi10(l1,l2,l3):
            return 27*l1*l2*l3
            
        phi =  [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10]

        #Gradients of basis functions in barycenter coordinates
        def g1(l1,l2,l3):
            return np.array([-0.5 * (27*l1*l1 - 18*l1 + 2), -0.5 * (27*l1*l1 - 18*l1 + 2)])
        def g2(l1,l2,l3):
            return np.array([0, 0.5 * (27*l2*l2 - 18*l2 + 2)])
        def g3(l1,l2,l3):
            return np.array([0.5 * (27*l3*l3 - 18*l3 + 2), 0])
            
        def g4(l1,l2,l3):
            return np.array([4.5 * (l2 * (6*l1 - 1) * (-1.0) + l1 * (3*l1 - 1) * 0.0), 4.5 * (l2 * (6*l1 - 1) * (-1.0) + l1 * (3*l1 - 1) * 1.0)])
        def g5(l1,l2,l3):
            return np.array([4.5 * (l2 * (3*l2 - 1) * (-1.0) + l1 * (6*l2 - 1) * 0.0), 4.5 * (l2 * (3*l2 - 1) * (-1.0) + l1 * (6*l2 - 1) * 1.0)])
        def g6(l1,l2,l3):
            return np.array([4.5 * (l3 * (6*l2 - 1) * 0.0 + l2 * (3*l2 - 1) * 1.0), 4.5 * (l3 * (6*l2 - 1) * 1.0 + l2 * (3*l2 - 1) * 0.0)])
        def g7(l1,l2,l3):
            return np.array([4.5 * (l3 * (3*l3 - 1) * 0.0 + l2 * (6*l3 - 1) * 1.0), 4.5 * (l3 * (3*l3 - 1) * 1.0 + l2 * (6*l3 - 1) * 0.0)])
        def g8(l1,l2,l3):
            return np.array([4.5 * (l1 * (6*l3 - 1) * 1.0 + l3 * (3*l3 - 1) * (-1.0)), 4.5 * (l1 * (6*l3 - 1) * 0.0 + l3 * (3*l3 - 1) * (-1.0))])
        def g9(l1,l2,l3):
            return np.array([4.5 * (l1 * (3*l1 - 1) * 1.0 + l3 * (6*l1 - 1) * (-1.0)), 4.5 * (l1 * (3*l1 - 1) * 0.0 + l3 * (6*l1 - 1) * (-1.0))])
            
        def g10(l1,l2,l3):
            return np.array([27.0 * (-l2*l3 + -l1*l3 + l1*l2 * 0.0), 27.0 * (-l2*l3 + l1*(-l3) + l1*l2 * 1.0)])
            
        grad = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        
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

    def Square_Mesh(self,Nx, Ny, x_i, x_f, y_i, y_f):
        #Constructs a square mesh for FEM
        x = np.linspace(x_i, x_f, Nx)
        y = np.linspace(y_i, y_f, Ny)

        T = np.zeros((3,2,2*(Nx-1)*(Ny-1)))
        for n in range(Ny-1):
            for m in range(Nx-1):
                T[0,:,2*(m+(Nx-1)*n)] = np.array([x[m],y[n]]) 
                T[1,:,2*(m+(Nx-1)*n)] = np.array([x[m],y[n+1]])
                T[2,:,2*(m+(Nx-1)*n)] = np.array([x[m+1],y[n]]) 
                
                T[0,:,2*(m+(Nx-1)*n)+1] = np.array([x[m+1],y[n+1]]) 
                T[1,:,2*(m+(Nx-1)*n)+1] = np.array([x[m],y[n+1]])
                T[2,:,2*(m+(Nx-1)*n)+1] = np.array([x[m+1],y[n]]) 

        #Tethers the global degrees of freedom to the local basis functions
        GDOF = 9*(Nx-1)*(Ny-1) +3*(Nx-1)+3*(Ny-1)+1
        X = np.linspace(x_i, x_f, 3*Nx - 2)
        Y = np.linspace(y_i, y_f, 3*Ny - 2)
        V = np.zeros((GDOF,2))
        for m in range(3*Ny-2):
            for n in range(3*Nx-2):
                V[n+(3*(Nx-1)+1)*m,:] = np.array([X[n],Y[m]]) 

        TV = np.zeros((GDOF,2*(Nx-1)*(Ny-1),1))
        for m in range(GDOF):
            for n in range(2*(Nx-1)*(Ny-1)):
                xi = T[0,:,n]
                xj = T[1,:,n]
                xk = T[2,:,n]
                l1,l2,l3 = self.Barycenter_Coords(xi,xj,xk)[:3]
                if l1(V[m,0],V[m,1])>=0 and l2(V[m,0],V[m,1])>=0 and l3(V[m,0],V[m,1])>=0:
                    if np.isclose(l1(V[m,0],V[m,1]),0.0):
                        if np.isclose(l2(V[m,0],V[m,1]),0.0):
                            TV[m,n,0] = 3
                            
                        elif np.isclose(l3(V[m,0],V[m,1]),0.0):
                            TV[m,n,0] = 2

                        elif l3(V[m,0],V[m,1]) > l2(V[m,0],V[m,1]):
                            TV[m,n,0] = 7
                            
                        elif l2(V[m,0],V[m,1]) > l3(V[m,0],V[m,1]):
                            TV[m,n,0] = 6

                    if np.isclose(l2(V[m,0],V[m,1]),0.0):
                        if np.isclose(l3(V[m,0],V[m,1]),0.0):
                            TV[m,n,0] = 1
                            
                        elif l3(V[m,0],V[m,1]) > l1(V[m,0],V[m,1]):
                            TV[m,n,0] = 8
                            
                        elif l1(V[m,0],V[m,1]) > l3(V[m,0],V[m,1]):
                            TV[m,n,0] = 9
                            
                    if np.isclose(l3(V[m,0],V[m,1]),0.0):
                        if l2(V[m,0],V[m,1]) > l1(V[m,0],V[m,1]):
                            TV[m,n,0] = 5
                            
                        elif l1(V[m,0],V[m,1]) > l2(V[m,0],V[m,1]):
                            TV[m,n,0] = 4
                            
                    else:
                        TV[m,n,0] = 10
                
                    
        return T, TV      
        
    def StiffnessMatrix(self,Nx, Ny, x_i, x_f, y_i, y_f):    
        #Construct stiffness matrix in FEM
        T, TV = self.Square_Mesh(Nx, Ny, x_i, x_f, y_i, y_f)
        GDOF = 9*(Nx-1)*(Ny-1) +3*(Nx-1)+3*(Ny-1)+1
        A = np.zeros((GDOF,GDOF))
        x = np.linspace(x_i, x_f, Nx)
        y = np.linspace(y_i, y_f, Ny)
        for n in range(GDOF):
            for m in range(GDOF):
                for j in range(T.shape[2]):
                    if TV[n,j,0] !=0 and TV[m,j,0]!= 0:
                        xi = T[0,:,j]
                        xj = T[1,:,j]
                        xk = T[2,:,j]
                        J, a = self.Barycenter_Coords(xi,xj,xk)[3:]
                        phi, grad = self.Basis_Functions(xi,xj,xk)
                        def f(l1,l2,l3):
                            return a*grad[int(TV[n,j,0]-1)](l1,l2,l3)@J.T@J@grad[int(TV[m,j,0]-1)](l1,l2,l3)/(xi[0]*l1 +xj[0]*l2+xk[0]*l3)
                        A[n,m] += self.IntegrateTriangle(f)
        return A   
