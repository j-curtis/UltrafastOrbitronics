### Code to compute solve Bloch equations of motion for two-sublattice model with spin and orbital
### Jonathan Curtis
### 08/04/2024

import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt 
import time 


### We will create a class to handle the simulations 
class simulation:

	### Generates a random set of spin orbital angles
	@staticmethod
	def random_angles():

		### This function will generate a random set of 12 angles
		### We distribute as follows  
		### The two sets of spin angles (4 total) are randomly chosen across the Bloch sphere 
		### Similarly for the angles parameterizing orb_u
		### The angles parameterizing chi are uniformly distributed as are eta 
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

	### Maps set of spin orbital angles to a cartesian vector 
	@staticmethod
	def angles_to_vector(angles):

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

	### Initialize method
	def __init__(self,J0,eta,soc,cf,hext):
		### Default will be to initialize to random angles 
		angls = simulation.random_angles()
		self.vectors = simulation.angles_to_vector(angls)

		self.J0 = J0
		self.eta = eta 

		### We now compute the s,p,d channel exchanges based on these values using the static function 
		self.JS = self.J0/(1.+2.*self.eta)
		self.JP = self.J0/(1.-3.*self.eta)
		self.JD = self.J0/(1.-self.eta)

		### And we further compute the J and K interactions in the symmetric and antisymmetric channels
		self.J0 = (self.JD -self.JS)/3. - self.JP
		self.J1 = (self.JD + self.JP)/2.
		self.J2 = self.JD/6. + self.JS/3. + self.JP/2.

		self.J1 = self.J1 - self.J2/2.

		self.K1 = self.JP/2. - self.JD/2. 
		self.K2 = -self.JD/6. - self.JS/3. +self.JP/2.

		self.K1 = self.K1 - self.K2/2.

		self.soc = soc
		self.cf = cf ### This should have shape of (2,3,3) and is the CF matrix on each site
		
		### External magnetic field 
		self.hext = hext

		self.energy = self.calc_energy(angls)

	### Now we add a method which will compute the energy of the given set of orbital angles for the current parameters 
	def calc_energy(self,angles):
		e = 0.
		vec = simulation.angles_to_vector(angles)

		for i in range(2):
			s = vec[9*i:(9*i +3)]
			u = vec[(9*i+3):(9*i+6)]
			v = vec[(9*i+6):(9*i+9)]

			e += self.soc*2.*s@np.cross(u,v) 
			e += u@self.cf[i]@u + v@self.cf[i]@v
			
			e += -self.hext@s

		spin_parity = vec[:3]@vec[9:12] + 0.25 

		L_sl = 2.*np.array([ np.cross(vec[3:6],vec[6:9]), np.cross(vec[12:15],vec[15:18])]) 
		l1l2 = L_sl[0]@L_sl[1] ### Angular momentum overlap

		Q_sl = [ np.outer(vec[3:6],vec[3:6]) +np.outer(vec[6:9],vec[6:9]) , np.outer(vec[12:15],vec[12:15]) +np.outer(vec[15:18],vec[15:18])  ]
		q1q2 = np.trace( Q_sl[0]@Q_sl[1])  ### Orbital polarization overlap

		### These are all twice the super exchange energy because they should be counted once for each sublattice in the system (not including z factor)
		se_j_energy = spin_parity*( self.J0 + self.J1*l1l2 + self.J2 *( 1. + q1q2) )
		se_k_energy = 0.5*( self.K1*l1l2 + self.K2*(1.+q1q2) )

		e += se_j_energy + se_k_energy

		return e

	### This will find the ground state for a given set of parameters
	### It will do so by variationally minimizing the energy over spin orbital angles 
	### It will save the new state over the old one and update the energy 
	def find_GS(self):
		a0 = simulation.random_angles()
		f = lambda a: self.calc_energy(a)
		
		### We should tweak the basin hopping temperature a bit to make sure we don't get stuck in a metastable state
		### We will take it to be about comparable to 70% of J0 
		temp = 0.7 * self.J0
		
		sol = opt.basinhopping(f,a0,T = temp)

		### Now we extract the set of angles and the energy 
		angles = sol.x 
		self.vectors = simulation.angles_to_vector(angles)
		self.energy = sol.fun 
		
		
	### This method returns the magnetization on each sublattice 
	def get_magnetic_state(self):
		m1 = self.vectors[0:3]
		m2 = self.vectors[9:12]
		
		return m1,m2

def main():
	J0 = 6.*10. ### We use meV and include coordination number here which, for cubic lattice is z = 6 
	eta = 0.1
	socs = np.linspace(0.,10.,20)

	cf = [ np.diag([20.,0.,-20.]), - np.diag([20.,0.,-20.]) ]
	
	hext = np.array([0.,0.,0.01])

	sim = simulation(J0,eta,0.,cf,hext)
	magzs = np.zeros_like(socs)

	for i in range(len(socs)):
		soc = socs[i]
		simulation.soc = soc
		sim.find_GS()
		m1,m2 = sim.get_magnetic_state()
		magzs[i] = m1[2]
		
	plt.plot(socs,magzs)
	plt.show()
	

if __name__ == "__main__":
	main()











