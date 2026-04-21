import numpy as np
import matplotlib.pyplot as plt

class Vlasov_Poisson1D():

    def __init__(self, alpha,k):
        self.alpha  = alpha
        self.k = k
        
    def initial_condition(self,f0,E0):
        self.f0 = f0
        self.E0 = E0
        
    def advect_z(self,f,v,z,dt):
        dz = z[1]-z[0]
        nu = dt*v/(dz)
        Up = np.zeros((z.size,v.size))
        Um = np.zeros((z.size,v.size))
        L1 = np.zeros((z.size,v.size))
        L2 = np.zeros((z.size,v.size))
        fp2 = np.roll(f,-2, axis=0)
        fp1 = np.roll(f,-1, axis=0)
        fm1 = np.roll(f, 1, axis=0)
        fm2 = np.roll(f, 2, axis=0)
        fmax = np.maximum.reduce([fp1,fm1,np.fmin(2*fm1-fm2,2*f-fp1),np.fmin(2*f-fm1, 2*fp1-fp2)])
        fmin = np.minimum.reduce([fp1,fm1,np.fmax(2*fm1-fm2,2*f-fp1),np.fmax(2*f-fm1, 2*fp1-fp2)])
        
        L1 = np.where(fp1>= f, np.minimum(2*(f - fmin),fp1 - f), -np.minimum(2*(fmax - f),(f - fp1)))
        L2 = np.where(f>= fm1, np.minimum(2*(fmax - f),f - fm1), -np.minimum(2*(f - fmin),(fm1 - f)))

        Up = np.where(v>=0, nu*(f + L1*(1-nu)*(2-nu)/6 +L2*(1-nu)*(1+nu)/6),0)
        Um = np.where(v<0 , nu*(fp1 -np.roll(L2,-1,axis= 0)*(1+nu)*(2+nu)/6 - np.roll(L1,-1,axis=0)*(1-nu)*(1+nu)/6),0)

        f = np.where(v>= 0, f + np.roll(Up, 1, axis=0) - Up, f + np.roll(Um, 1, axis=0) - Um)

        return f  
        
    def advect_v(self,f,v,z,dt,E):
        dv = v[1]-v[0]
        nu = dt*E/(dv)
        Up = np.zeros((z.size,v.size))
        Um = np.zeros((z.size,v.size))
        L1 = np.zeros((z.size,v.size)) 
        L2 = np.zeros((z.size,v.size))
        fp1 = np.zeros_like(f)
        fp1[:, :-1] = f[:, 1:]
        fp2 = np.zeros_like(f)
        fp2[:, :-2] = f[:, 2:]
        fm1 = np.zeros_like(f)
        fm1[:,1:] = f[:,:-1]
        fm2 = np.zeros_like(f)
        fm2[:,2:] = f[:,:-2]
        fmax = np.maximum.reduce([fp1,fm1,np.fmin(2*fm1-fm2,2*f-fp1),np.fmin(2*f-fm1, 2*fp1-fp2)])
        fmin = np.minimum.reduce([fp1,fm1,np.fmax(2*fm1-fm2,2*f-fp1),np.fmax(2*f-fm1, 2*fp1-fp2)])

        L1 = np.where(fp1>= f, np.minimum(2*(f - fmin),fp1 - f), -np.minimum(2*(fmax - f),(f - fp1)))
        L2 = np.where(f>= fm1, np.minimum(2*(fmax - f),f - fm1), -np.minimum(2*(f - fmin),(fm1 - f)))
        L1p1 = np.zeros_like(L1)
        L1p1[:,:-1] = L1[:,1:]
        L2p1 = np.zeros_like(L2)
        L2p1[:,:-1] = L2[:,1:]

        Up = np.where(nu>=0, nu[:, None]*(f + L1*(1-nu[:,None])*(2-nu[:,None])/6 +L2*(1-nu[:,None])*(1+nu[:,None])/6),0)
        Um = np.where(nu< 0, nu[:, None]*(fp1 - L2p1*(1+nu[:,None])*(2+nu[:,None])/6 -L1p1*(1-nu[:,None])*(1+nu[:,None])/6 ),0 )

        Upm1 = np.zeros_like(Up) 
        Upm1[:,1:] = Up[:,:-1]
        Umm1 = np.zeros_like(Um) 
        Umm1[:,1:] = Um[:,:-1]
        
        fnew = np.where(nu>= 0, f + Upm1  - Up, f + Umm1 - Um)

        fnew[:, :2] = 0.0
        fnew[:, -2:] = 0.0
        return fnew

    def Poisson_solve(self,z,v,f):
        dv = v[1]-v[0]
        rho = np.sum(f, axis=1) * dv
        rho_tilde = rho - rho.mean()

        rho_k = np.fft.fft(rho_tilde)
        E_k = np.zeros_like(rho_k, dtype=complex)

        mask = self.k != 0.0
        E_k[mask] = -1j * self.alpha * rho_k[mask] / self.k[mask]
        E_k[~mask] = 0.0  # enforce zero-mean field

        E = np.fft.ifft(E_k).real
        E -= E.mean()
        return E
        
    def solve(self,N,dt,z,v):
        f = self.f0.copy()
        E = np.zeros(z.size)
        E[0] = self.E0
        for i in range(N):
            f = self.advect_z(f,v,z,dt/2)
            E = self.Poisson_solve(z,v,f)
            f = self.advect_v(f,v,z,dt,E)
            f = self.advect_z(f,v,z,dt/2)
            if not np.isfinite(f).all():
                print(f"non-finite values at step {i}")
                print("min/max f:", np.nanmin(f), np.nanmax(f))
                print("min/max E:", np.nanmin(E), np.nanmax(E))
                break
        return f, E
        
Nz = 256
Nv = 256
z = np.linspace(0.0, 2.0 * np.pi, Nz,  endpoint=False)
v = np.linspace(-5, 5, Nv)
Z, V = np.meshgrid(z, v, indexing="ij")

# Two Maxwellian beams in velocity
base = 0.5 *3.3* (
    np.exp(-(3.3**2) * (V + 1) ** 2) +
    np.exp(-(3.3**2) * (V - 1) ** 2)
)/(np.sqrt(np.pi))

# Small spatial perturbation to seed instability
eps = 1e-2
k0 = 1.0
f0 = base * (1.0 + eps * np.cos(k0 * Z))

N = 1200
dt = 0.01

problem = Vlasov_Poisson1D(3, 2.0 * np.pi * np.fft.fftfreq(Nz, d=z[1]-z[0]))
problem.initial_condition(f0,0)
f,E = problem.solve(N,dt,z,v)

plt.pcolormesh(Z, V, f, shading='auto')
plt.colorbar(label="Magnitude")
plt.xlabel("z")
plt.ylabel("v")
plt.title("Distribution Function")
plt.show() 
