import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


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
            return 4.5*l1*l2*(3*l2-1)
        def phi5(l1,l2,l3):
            return 4.5*l1*l2*(3*l1-1)
        def phi6(l1,l2,l3):
            return 4.5*l2*l3*(3*l3-1)
        def phi7(l1,l2,l3):
            return 4.5*l2*l3*(3*l2-1)
        def phi8(l1,l2,l3):
            return 4.5*l1*l3*(3*l1-1)
        def phi9(l1,l2,l3):
            return 4.5*l1*l3*(3*l3-1)
            
        def phi10(l1,l2,l3):
            return 27*l1*l2*l3
            
        phi =  [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10]

        #Gradients of basis functions in barycenter coordinates
        def g1(l1,l2,l3):
            return np.array([-0.5 * (27*l1*l1 - 18*l1 + 2), -0.5 * (27*l1*l1 - 18*l1 + 2)])
        def g2(l1,l2,l3):
            return np.array([0.5 * (27*l2*l2 - 18*l2 + 2),0])
        def g3(l1,l2,l3):
            return np.array([0,0.5 * (27*l3*l3 - 18*l3 + 2)])

        def g4(l1,l2,l3):
            return np.array([4.5 * (l2 * (3*l2 - 1) * (-1.0) + l1 * (6*l2 - 1) * 1.0),4.5 * (l2 * (3*l2 - 1) * (-1.0) + l1 * (6*l2 - 1) * 0.0)])
        def g5(l1,l2,l3):
            return np.array([4.5 * (l2 * (6*l1 - 1) * (-1.0) + l1 * (3*l1 - 1) * 1.0),4.5 * (l2 * (6*l1 - 1) * (-1.0) + l1 * (3*l1 - 1) * 0.0)])
        def g6(l1,l2,l3):
            return np.array([ 4.5 * (l3 * (3*l3 - 1) * 1.0 + l2 * (6*l3 - 1) * 0.0),4.5 * (l3 * (3*l3 - 1) * 0.0 + l2 * (6*l3 - 1) * 1.0)])
        def g7(l1,l2,l3):
            return np.array([ 4.5 * (l3 * (6*l2 - 1) * 1.0 + l2 * (3*l2 - 1) * 0.0),4.5 * (l3 * (6*l2 - 1) * 0.0 + l2 * (3*l2 - 1) * 1.0)])
        def g8(l1,l2,l3):
            return np.array([4.5 * (l1 * (3*l1 - 1) * 0.0 + l3 * (6*l1 - 1) * (-1.0)),4.5 * (l1 * (3*l1 - 1) * 1.0 + l3 * (6*l1 - 1) * (-1.0))])
        def g9(l1,l2,l3):
            return np.array([ 4.5 * (l1 * (6*l3 - 1) * 0.0 + l3 * (3*l3 - 1) * (-1.0)),4.5 * (l1 * (6*l3 - 1) * 1.0 + l3 * (3*l3 - 1) * (-1.0))])
        
        def g10(l1,l2,l3):
            return np.array([27.0 * (-l2*l3 + l1*l3 ), 27.0 * (-l2*l3  + l1*l2 * 1.0)])
            
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
                T[1,:,2*(m+(Nx-1)*n)] = np.array([x[m+1],y[n]])
                T[2,:,2*(m+(Nx-1)*n)] = np.array([x[m],y[n+1]]) 
                
                T[0,:,2*(m+(Nx-1)*n)+1] = np.array([x[m+1],y[n]]) 
                T[1,:,2*(m+(Nx-1)*n)+1] = np.array([x[m+1],y[n+1]])
                T[2,:,2*(m+(Nx-1)*n)+1] = np.array([x[m],y[n+1]]) 

        #Tethers the global degrees of freedom to the local basis functions
        X = np.linspace(x_i, x_f, 3*Nx - 2)
        Y = np.linspace(y_i, y_f, 3*Ny - 2)
        V = {}
        boundary_points = {}
        for m in range(3*Ny-2):
            for n in range(3*Nx-2):
                V[(round(float(X[n]),12),round(float(Y[m]),12))] = n+(3*(Nx-1)+1)*m
                if X[n] == float(x_i) or X[n] == float(x_f) or Y[m] == float(y_i) or Y[m] == float(y_f):
                    boundary_points[(round(float(X[n]),12),round(float(Y[m]),12))] = n+(3*(Nx-1)+1)*m
                
                
        TV = np.zeros((2*(Nx-1)*(Ny-1), 10))
        bary_basis = np.array([ [1.0,   0.0,   0.0], [0.0,   1.0,   0.0], [0.0,   0.0,   1.0],
                                [2/3,   1/3,   0.0], [1/3,   2/3,   0.0], [0.0,   2/3,   1/3],
                                [0.0,   1/3,   2/3], [1/3,   0.0,   2/3], [2/3,   0.0,   1/3],
                                [1/3,   1/3,   1/3]])
        for m in range(T.shape[2]):
            xi = T[0,:,m]
            xj = T[1,:,m]
            xk = T[2,:,m]
            l1,l2,l3 = self.Barycenter_Coords(xi,xj,xk)[:3]
            for n in range(10):
                p = bary_basis[n,0]*xi + bary_basis[n,1]*xj + bary_basis[n,2]*xk
                TV[m,n] = V[(round(float(p[0]), 12),round(float(p[1]), 12))]

        
        return T, V, TV, boundary_points  
        
    def StiffnessMatrix(self,Nx, Ny, x_i, x_f, y_i, y_f, T,V,TV):    
        #Construct stiffness matrix in FEM
        GDOF = 9*(Nx-1)*(Ny-1) +3*(Nx-1)+3*(Ny-1)+1
        A = lil_matrix((GDOF,GDOF))
        for n in range(T.shape[2]):
            xi = T[0,:,n]
            xj = T[1,:,n]
            xk = T[2,:,n]
            J, a = self.Barycenter_Coords(xi,xj,xk)[3:]
            phi, grad = self.Basis_Functions(xi,xj,xk)
            B = TV[n,:]
            for j in range(10):
                for k in range(10):
                    def f(l1,l2,l3):
                        return abs(a)*grad[j](l1,l2,l3)@J.T@J@grad[k](l1,l2,l3)/(xi[0]*l1 +xj[0]*l2+xk[0]*l3)
                    A[int(B[j]),int(B[k])] += self.IntegrateTriangle(f) 
        return A 
    def LoadVector(self,Nx, Ny, x_i, x_f, y_i, y_f, Curr, T,V,TV):  
        #Construct load vector in FEM
        GDOF = 9*(Nx-1)*(Ny-1) +3*(Nx-1)+3*(Ny-1)+1
        b = np.zeros(GDOF)
        for n in range(T.shape[2]):
            xi = T[0,:,n]
            xj = T[1,:,n]
            xk = T[2,:,n]
            J, a = self.Barycenter_Coords(xi,xj,xk)[3:]
            phi, grad = self.Basis_Functions(xi,xj,xk)
            B = TV[n,:]
            for j in range(10):
                def f(l1,l2,l3):
                    return abs(a)*phi[j](l1,l2,l3)*Curr(xi[0]*l1 +xj[0]*l2+ xk[0]*l3,xi[1]*l1 +xj[1]*l2+ xk[1]*l3 )
                b[int(B[j])] += self.IntegrateTriangle(f) 
        return b
        
    def BoundaryConditions(self,boundary_points, A, b, u, boundary_values):
        #Imposes Dirichlet boundary conditions on all boundary nodes
        for key, n in boundary_points.items():
            u[n] = boundary_values[key]
            b[n] = boundary_values[key]
            A[n,:] = 0
            A[n,n] = 1
        return A, b, u
    def Solve(self,Nx, Ny, x_i, x_f, y_i, y_f):
        T, V, TV = self.Square_Mesh(Nx, Ny, x_i, x_f, y_i, y_f)[:3]
        A = self.StiffnessMatrix(Nx, Ny, x_i, x_f, y_i, y_f,T,V,TV)
        def Curr(x,y):
            return 10
        b = self.LoadVector(Nx, Ny, x_i, x_f, y_i, y_f, Curr,T,V,TV)
        GDOF = 9*(Nx-1)*(Ny-1) +3*(Nx-1)+3*(Ny-1)+1
        u = np.zeros(GDOF)
        boundary_points = self.Square_Mesh(Nx, Ny, x_i, x_f, y_i, y_f)[3]
        boundary_values = {}
        for key, n in boundary_points.items():
            boundary_values[key] = 0
        A, b, u = self.BoundaryConditions(boundary_points, A, b, u, boundary_values)
        A = A.tocsr()
        u = spsolve(A, b)
        def U(x,y):
            U = 0
            k=0
            for n in range(T.shape[2]):
                xi = T[0,:,n]
                xj = T[1,:,n]
                xk = T[2,:,n]
                l1,l2,l3 = self.Barycenter_Coords(xi,xj,xk)[:3]
                phi = self.Basis_Functions(xi,xj,xk)[0]
                if (l1(x,y)>= float(0)) and (l2(x,y)>= float(0)) and (l3(x,y)>= float(0)):
                    k+=1
                    for j in range(10):
                        U += u[int(TV[n,j])]*phi[j](l1(x,y),l2(x,y),l3(x,y))
                if k==1:
                    break
            return U  
            
        return u,U
problem = Finite_Element_Solver()
u, U = problem.Solve(10, 10, 1, 2, 1, 2)
r = np.linspace(1, 2, 3*10-2)
z = np.linspace(1, 2, 3*10-2)
R, Z = np.meshgrid(r, z, indexing="ij")
U_grid = np.vectorize(U)
plt.pcolormesh(R, Z, U_grid(R,Z), shading='auto')
plt.colorbar(label="Magnitude")
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Magnetic Flux Label")
plt.show() 
