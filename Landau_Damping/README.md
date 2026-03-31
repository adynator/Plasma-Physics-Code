Landau Damping Project

This model considers a (1+1)d model of a homogeneous plasma perturbed about a Maxwellian equilibrium\
$$\frac{\partial g}{\partial t}+ v\frac{\partial g}{\partial z}+ v F_{0}\frac{\partial \varphi }{\partial z} = C[g]$$

where $g$ is the perturbed distribution function, $F_{0}$ is the Maxwellian equilibrium, $\varphi$ is the electrostatic potential, and $C[g]$ is the linearised Lenard-Bernstein collision operator. In this problem, we expand $g$ in a basis of Hermite polynomials in velocity space and Fourier modes in position space. We are left with a coupled system of first-order ODEs, which we solve using the backward Euler method and the Thomas algorithm to invert the tridiagonal matrix present in the problem. We plot the potential and observe Landau damping in the potential as it exponentially decreases. In the collisionless regime, the free energy stays constant, while when collisions are turned on, the free energy also decays to zero. The decrease in the free energy when collisions are present signals that the magnitude of $g$ decreases over time. The plot of $g$ demonstrates this phenomenon as well as phase mixing. 
