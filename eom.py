### Code to compute solve Bloch equations of motion for two-sublattice model with spin and orbital
### Jonathan Curtis
### 08/04/2024

import numpy as np
from scipy import optimize as opt
from scipy import integrate as intgrt
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
			phi = np.random.ranf()*2.*np.pi
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
	def __init__(self,J0,J1,J2,K1,K2,soc,cf,hext):
		### Default will be to initialize to random angles 
		angls = simulation.random_angles()
		self.vectors = simulation.angles_to_vector(angls)

		self.J0 = J0
		self.J1 = J1
		self.J2 = J2

		self.K1 = K1
		self.K2 = K2

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
		temp = 0.1 * np.abs(self.J0)
		
		sol = opt.basinhopping(f,a0,T = temp)

		### Now we extract the set of angles and the energy 
		angles = sol.x 
		self.vectors = simulation.angles_to_vector(angles)
		self.energy = sol.fun 
				
	### This method returns the magnetization on each sublattice 
	def get_spin_mag(self):
		m1 = self.vectors[0:3]
		m2 = self.vectors[9:12]
		
		return m1,m2

	### This method returns the orbital magnetization on each sublattice 
	def get_orbital_mag(self):
		u1 = self.vectors[3:6]
		v1 = self.vectors[6:9]
		u2 = self.vectors[12:15]
		v2 = self.vectors[15:18]

		l1 = 2.*np.cross(u1,v1)
		l2 = 2.*np.cross(u2,v2)
		
		return l1,l2
		
	
	#################################
	### Now we implement dyanmics ###	
	#################################
	
	### This will set the time parameters of the dynamics
	### Pass an array of time points
	def set_dynamics_times(self,times):
		self.times = times
		self.t0 = self.times[0]
		self.tf = self.times[-1]
		self.ntimes = len(self.times)
		
		### We also set an array for the state vectors at each time point, which may be quite large 
		self.vector_dynamics = np.zeros((18,self.ntimes))
	
	### This function will be the analytical expression of the magnetic field as a function of time 
	### We will pass a reference to a function 
	def set_dynamics_hext(self,f):
		self.hext_t = f
		
	### Now we construct the equations of motion function 
	def eom_function(self,t,X):
		dXdt = np.zeros(18)
		
		lvec = np.zeros((2,3)) ### This will be the angular momentum on each sublattice
		qtensor = np.zeros((2,3,3)) ### This will be the angular momentum nematic tensor on each sublattice
		spin = np.zeros((2,3)) ### This is the spin on each sublattice, used for the spin parity operator 
		
		### Unpack all variables
		for i in range(2):
			s = X[9*i:(9*i +3)]
			u = X[(9*i+3):(9*i+6)]
			v = X[(9*i+6):(9*i+9)]
			
			lvec[i,:] = 2.*np.cross(u,v)
			qtensor[i,:,:] = np.outer(u,u) + np.outer(v,v)
			spin[i,:] = s

		for i in range(2):
			### CF terms in EOM
			dXdt[(9*i+3):(9*i+6)] += self.cf[i]@X[(9*i+6):(9*i+9)]
			dXdt[(9*i+6):(9*i+9)] += -self.cf[i]@X[(9*i+3):(9*i+6)]

			### SOC terms in EOM
			dXdt[9*i:(9*i +3)] += self.soc*np.cross(lvec[i,:],spin[i,:])
			dXdt[(9*i+3):(9*i+6)] += self.soc*np.cross(spin[i,:],X[(9*i+3):(9*i+6)] )
			dXdt[(9*i+6):(9*i+9)] += self.soc*np.cross(spin[i,:],X[(9*i+6):(9*i+9)] )


			### external field in EOM
			h_t = self.hext + self.hext_t(t)
			dXdt[9*i:(9*i +3)] += np.cross(-h_t,spin[i,:])

			### KK terms in EOM
			jeff = self.J0 + self.J1 * lvec[0,:]@lvec[1,:] + self.J2 *(1. +  np.trace(qtensor[0,...]@qtensor[1,...]) )
			dXdt[9*i:(9*i +3)] += jeff*np.cross(spin[i-1,:],spin[i,:])


			spin_parity = spin[0,:]@spin[1,:] + 0.25
			
			### U and V EOM
			xieff = (spin_parity*self.J1 + 0.5*self.K1 )*lvec[i-1,:]
			heff = (spin_parity*self.J2 + 0.5*self.K2 )*( np.eye(3) - qtensor[i-1,:,:] )
			
			dXdt[(9*i+3):(9*i+6)] +=  heff @ X[(9*i+6):(9*i+9)] + np.cross(xieff,X[(9*i+3):(9*i+6)])
			dXdt[(9*i+6):(9*i+9)] += -heff @ X[(9*i+3):(9*i+6)] + np.cross(xieff,X[(9*i+6):(9*i+9)])
			
			
		### We have finished constructing the EOM function
		return dXdt
		
	### This function will solve the equations of motion starting from the ground state and store the simulation output in the correct variables
	def run_dynamics(self):
		sol = intgrt.solve_ivp( self.eom_function,(self.t0,self.tf),self.vectors,t_eval = self.times)
		self.vector_dynamics = sol.y
		
	### This method returns the magnetization dynamics
	def get_spin_mag_t(self):
		m1 = self.vector_dynamics[0:3,:]
		m2 = self.vector_dynamics[9:12,:]
		
		return m1,m2

	### This method returns the orbital magnetization on each sublattice 
	def get_orbital_mag_t(self):
		u1 = self.vector_dynamics[3:6,:]
		v1 = self.vector_dynamics[6:9,:]
		u2 = self.vector_dynamics[12:15,:]
		v2 = self.vector_dynamics[15:18,:]

		l1 = 2.*np.cross(u1,v1,axis=0)
		l2 = 2.*np.cross(u2,v2,axis=0)
		
		return l1,l2

	### This method returns the orbital quadrupolar tensors
	def get_orbital_quad_t(self):
		u1 = self.vector_dynamics[3:6,:]
		v1 = self.vector_dynamics[6:9,:]
		u2 = self.vector_dynamics[12:15,:]
		v2 = self.vector_dynamics[15:18,:]

		q1 = np.zeros((3,3,self.ntimes))
		q2 = np.zeros((3,3,self.ntimes))

		for i in range(3):
			for j in range(3):
				q1[i,j,:] = u1[i,:]*u1[j,:] + v1[i,:]*v1[j,:]
				q2[i,j,:] = u2[i,:]*u2[j,:] + v2[i,:]*v2[j,:]
		
		return q1,q2
		

def main():
	J0 = -1.
	J1 = -.3
	J2 = 0.23
	K1 = 0.3
	K2 = 0.9
	soc = 0.6

	cf = [ np.diag([0.3,0.,-0.3]), - np.diag([0.3,0.,-0.3])]

	hext = np.array([0.,0.,0.05])

	t0 = time.time()
	
	sim = simulation(J0,J1,J2,K1,K2,soc,cf,hext)
	sim.find_GS()
	print(sim.J0)

	times = np.linspace(0.,100.,300)
	hext_dynamics = lambda t : np.array([0.,0.,0.])
	
	sim.set_dynamics_times(times)
	sim.set_dynamics_hext(hext_dynamics)
	sim.run_dynamics()
	
	m1,m2 = sim.get_spin_mag_t()

	for i in range(3):
		plt.plot(sim.times,m1[i,:])
		plt.show()
		plt.plot(sim.times,m2[i,:])
		plt.show()
	
	l1,l2 = sim.get_orbital_mag_t()

	for i in range(3):
		plt.plot(sim.times,l1[i,:])
		plt.show()
		plt.plot(sim.times,l2[i,:])
		plt.show()

	q1,q2 = sim.get_orbital_quad_t()

	for i in range(3):
		for j in range(3):
			plt.plot(sim.times,q1[i,j,:])
			plt.show()
			plt.plot(sim.times,q2[i,j,:])
			plt.show()

	t1 = time.time()
	print("total time: ",t1-t0,"s")




if __name__ == "__main__":
	main()











