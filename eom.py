### Code to compute solve Bloch equations of motion for two-sublattice model with spin and orbital
### Jonathan Curtis
### 08/04/2024

import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt 
import time 

#######################################
### Ground state variational methods
#######################################

### This will convert a set of angles into vectors
### We encode the angles as follows
### There are 12 angles
### [0:6] are for first sublattice and [6:12] are for second sublattice
### Within each sublattice we have 
### [0:2] for spin vector
### [2:4] for orbital u vector 
### [4:6] for orbital v vector as 

### We then return in the Cartesian representation which has 18 components
### These are organized as [0:9] and [9:18] for sublattice 1 and 2
### In each sublattice we organize as 
### [0:3] = spin
### [3:6] = orbital u
### [6:9] = orbital v
def angles_to_vector(angles):

	angles_sl = [angles[:6],angles[6:]]

	vectors_sl = np.zeros((2,9)) ### 0-3 = spin, 3-6 = u, 6-9 = v

	for i in range(2):
		theta = angles_sl[i][0] ### Spin polar angle
		phi = angles_sl[i][1] ### Spin azimuthal angle

		alpha = angles_sl[i][2] ### Orbital u polar angle
		beta = angles_sl[i][3] ### Orbital u azimuthal angle

		chi = angles_sl[i][4] ### Orbital angular momentum noncolinear angle
		eta = angles_sl[i][5] ### Orbital v angle in plane perpindicular to u 

		vectors_sl[i,:3] = 0.5*np.array([ np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta) ])

		vectors_sl[i,3:6] = np.cos(chi)*np.array([ np.cos(beta)*np.sin(alpha), np.sin(beta)*np.sin(alpha), np.cos(alpha) ])

		### Orbital v must be constructed as sin(chi)*orthogonal to u 
		vectors_sl[i,6:9] = np.sin(chi)*np.cos(eta)*np.array([ -np.sin(beta), np.cos(beta),  0.]) + np.sin(chi)*np.sin(eta)*np.array([ np.cos(beta)*np.cos(alpha), np.sin(beta)*np.cos(alpha), -np.sin(alpha) ])

	return np.concatenate([vectors_sl[0,:],vectors_sl[1,:]])

### This function will generate a random set of 12 angles
### We distribute as follows  
### The two sets of spin angles (4 total) are randomly chosen across the Bloch sphere 
### Similarly for the angles parameterizing orb_u
### The angles parameterizing chi are uniformly distributed as are eta 
def random_angles():
	angles = np.zeros((2,6))

	for i in range(2):
		phi = np.random.ranf()*np.pi
		theta = np.arccos(2.*np.random.ranf()-1.) ### We take the arccos of a random value in [-1.1] -- this generates a random distribution on sphere 

		beta = phi = np.random.ranf()*np.pi
		alpha = np.arccos(2.*np.random.ranf()-1.) ### We take the arccos of a random value in [-1.1] -- this generates a random distribution on sphere 

		### Uniformly from the circle
		chi = np.random.ranf()*2.*np.pi
		eta = np.random.ranf()*2.*np.pi

		angles[i,0] = theta
		angles[i,1] = phi
		angles[i,2] = alpha
		angles[i,3] = beta
		angles[i,4] = chi
		angles[i,5] = eta

	return np.concatenate([angles[0,:],angles[1,:]])



#######################################
### Methods for evaluating Hamiltonian
#######################################

### This function will compute a set of superexchange constants given the overall J0 and the Hund's coupling for the model laid out in the notes
### First we compute the interactions in each of the intermediate state channels; S,P,D
### Then we compute the projections of these interactions onto the operators 1, L1.L2, and 1/2{L1,L1}. 1/2{L2,L2}
### We return these as a set of J0,J1,J2,K1,K2 (we drop K0 as it is a true constant)

def se_params(J0,eta):

	JS = J0/(1.+2.*eta)
	JP = J0/(1.-3.*eta)
	JD = J0/(1-eta)

	### Now we have to workout all of the projections onto each operator 
	### Check these values from the notes  --- these values are off memory at the moment
	J0 = (JD-JS)/3. - JP
	J1 = (JD + JP)/2.
	J2 = JD/6. + JS/3. + JP/2.

	J1 = J1 - J2/2. ### There is a contribution from the (L1.L2)^2 which renormalizes the J1 value when written in terms of the symmetrized operators 

	K1 = (-JD + JP)/2.
	K2 = -JD/6. - JS/3. + JP/2.

	K1 = K1 - K2/2. ### Same story for K operators

	return np.array([J0,J1,J2,K1,K2])


### This computes the energy due to an external magnetic field
def hext_energy(X,hext):
	### Given the state vector X we compute the energy for external field hext

	energy = 0.

	for i in range(2):
		s = X[9*i:(9*i +3)]

		energy += -hext@s

	return energy

### This computes the energy due to spin-orbit coupling
def soc_energy(X,soc):
	### Given the state vector X we compute the energy for spin-orbit coupling soc

	energy = 0.

	for i in range(2):
		s = X[9*i:(9*i +3)]
		u = X[(9*i+3):(9*i+6)]
		v = X[(9*i+6):(9*i+9)]

		energy += soc*2.*s@np.cross(u,v) 

	return energy

### This computes the energy due to crystal-field
def cf_energy(X,cf):
	### cf is an array of two 3x3 matrices which are symmetric

	energy = 0.

	for i in range(2):
		u = X[(9*i+3):(9*i+6)]
		v = X[(9*i+6):(9*i+9)]

		energy += u@(cf[i])@u + v@(cf[i])@v

	return energy

### This computes the energy due to superexchange 
def se_energy(X,J0,eta):

	### First we construct the superexchange constants
	se_vals = se_params(J0,eta)

	spin_parity = X[:3]@X[9:12] + 0.25

	L_sl = 2.*np.array([ np.cross(X[3:6],X[6:9]), np.cross(X[12:15],X[15:18])]) 

	l1l2 = L_sl[0]@L_sl[1] ### Angular momentum overlap

	Q_sl = [ np.outer(X[3:6],X[3:6]) +np.outer(X[6:9],X[6:9]) , np.outer(X[12:15],X[12:15]) +np.outer(X[15:18],X[15:18])  ]

	q1q2 = np.trace( Q_sl[0]@Q_sl[1])  ### Orbital polarization overlap

	### These are all twice the super exchange energy because they should be counted once for each sublattice in the system (not including z factor)
	se_j_energy = spin_parity*( se_vals[0] + se_vals[1]*l1l2 + se_vals[2] *( 1. + q1q2) )
	se_k_energy = 0.5*( se_vals[3]*l1l2 + se_vals[4]*(1.+q1q2) )

	return se_j_energy + se_k_energy


### This is the total energy function
def energy(X,J0,eta,soc,cf,hext):
	return soc_energy(X,soc) + cf_energy(X,cf) + se_energy(X,J0,eta) + hext_energy(X,hext)


### This method will find the ground state for a given set of parameters using basin hopping method from scipy
### It is important we minimize over angles, not vectors, as the vectors need to be constrained otherwise whereas the angles resolve this automatically
def find_GS(J0,eta,soc,cf,hext):
	a0 = random_angles()

	f = lambda a: energy(angles_to_vector(a), J0,eta,soc,cf,hext)

	sol = opt.basinhopping(f,a0)

	return sol

### This method will find the lowest energy given the spin states are held fixed
### We hold the angles which define the spins fixed every interation of the minimization
def find_GS_fixed_S(spin_angles,J0,eta,soc,cf,hext):
	a0 = random_angles()

	a0[:2] = spin_angles[:2]
	a0[6:8] = spin_angles[2:]

	def restricted_energy_function(a):
		a[:2] = spin_angles[:2]
		a[6:8] = spin_angles[2:]

		return energy(angles_to_vector(a), J0,eta,soc,cf,hext)

	sol = opt.basinhopping(restricted_energy_function,a0)

	return sol


### This method will compute the magnetization from the solution object returned from the find_GS method
def find_M(sol):
	angles = sol.x

	vec = angles_to_vector(angles)

	m1 = vec[:3]
	m2 = vec[9:12]

	return m1,m2

def main():
	J0 = 10. ### We use meV 
	eta = 0.1
	soc = 10.
	hext = np.array([ 0.,0.,0. ])

	cf = [ np.diag([10.,0.,-10.]), - np.diag([10.,0.,-10.]) ]

	nthetas = 20
	nphis = 20
	thetas = np.arccos(np.linspace(-1.,1.,nthetas))
	phis = np.linspace(0.,2.*np.pi,nphis)

	energies = np.zeros((nthetas,nphis))

	for i in range(nthetas):
		for j in range(nphis):
			theta = thetas[i]
			phi = phis[j]
			sol = find_GS_fixed_S([theta,phi,theta,phi],J0,eta,soc,cf,hext)
			energies[i,j] = sol.fun

	plt.imshow(energies)
	plt.colorbar()
	plt.show()



if __name__ == "__main__":
	main()











