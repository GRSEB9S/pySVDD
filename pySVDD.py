import numpy as np
import quadprog

class SVDD:
	def __init__(self, C, gamma):
		self.C = C
		self.gamma = gamma
		
	def fit(self, X, y):
		"""
		Fit the model according to the given training data.
		
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vector, where n_samples in the number of samples and
			n_features is the number of features.
		y : array-like, shape = [n_samples], optional
			Target vector relative to X
		sample_weight : array-like, shape = [n_samples], optional
			Array of weights that are assigned to individual
			samples. If not provided,
			then each sample is given unit weight.
			
		Returns
		-------
		self : object
			Returns self.
		"""
		
		# Find coefficients and support vectors
		P, q, G, h, A, b = self.build_qp(X, y)
		self.beta = y*self.quadprog_solve_qp(self._nearestPD(P), q, G, h, A, b)
		self.sv = X[abs(self.beta) > 1e-6,:]
		self.support_ = np.squeeze(np.where(abs(self.beta) > 1e-6))
		self.beta = self.beta[abs(self.beta) > 1e-6]
		
		# Find decision threshold
		R2, _ = self.radius(self.sv)
		self.threshold = np.mean(R2)
		
		return self
	
	def predict(self, X):
		y = np.sign(self.decision_function(X)).astype('int')
		y[y == 0] = 1
		return y
	
	def decision_function(self, X):
		radius, _ = self.radius(X)
		return self.threshold - radius
		
	def radius(self, z):
		kap = 0
		lam = 0
		mu = 0
		
		for i in range(len(self.beta)):
			Kxz, dKxz = self.rbf_kernel(self.sv[i,:], z)
			kap = kap + self.beta[i]*Kxz
			mu = mu + self.beta[i]*dKxz
			for j in range(len(self.beta)):
				Kxx, _ = self.rbf_kernel(self.sv[i,:], self.sv[j,:])
				lam = lam + self.beta[i]*self.beta[j]*Kxx
				
		R2 = 1 - 2*kap + lam
		dR2 = -2*mu
		
		return R2, dR2
		
	def rbf_kernel(self, x, z):
		if z.ndim > 1:
			K = np.exp(-self.gamma*np.linalg.norm(x - z, axis = 1)**2)
			#dK = 2*self.gamma*(x - z)*np.exp(-self.gamma*np.linalg.norm(x - z, axis = 1)**2)
			dK = np.dot(np.diag(np.exp(-self.gamma*np.linalg.norm(x - z, axis = 1)**2)), 2*self.gamma*(x - z))
		else:
			K = np.exp(-self.gamma*np.linalg.norm(x - z)**2)
			dK = 2*self.gamma*(x - z)*np.exp(-self.gamma*np.linalg.norm(x - z)**2)
		
		return K, dK
		
	def build_qp(self, X, y):
		
		# Build P
		P = np.eye(len(y))
		for i in range(len(y) - 1):
			for j in range(i + 1, len(y)):
				Kxx, _ = self.rbf_kernel(X[i,:],X[j,:])
				P[i,j] = y[i]*y[j]*Kxx
		P = self._nearestPD(P + P.T - np.eye(len(y)))
		
		# Build q
		q = np.zeros(len(y))
		
		# Build G
		G = np.vstack((-np.eye(len(y)),np.eye(len(y))))
		
		# Build h
		h = np.hstack((np.zeros(len(y)),self.C*np.ones(len(y))))
		
		# Build A
		A = y
		
		# Build b
		b = 1.
		
		return P, q, G, h, A, b
	
	def quadprog_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
		"""
		Quadratic programming solver interface.  Solves the following:
		
		min		0.5*x.T*P*x + q.T*x
		s.t.	G*x <= h
				A*x = b
				
		This is CVXOPT notation but we use quadprog.  This method translates
		to quadprog notation.
		
		Parameters
		----------
		P : array-like, shape = [n_samples,n_samples]
			Lower bound for bisection search
		q : number
			Label for lower bound
		G : array-like, shape = [n_,n_samples]
			Upper bound for bisection search
		h : number
			Label for upper bound
			
		Returns
		-------
		self : object
			Returns self.
		"""
		qp_G = .5 * (P + P.T)   # make sure P is symmetric
		qp_a = -q
		if A is not None:
			qp_C = -np.vstack([A, G]).T
			qp_b = -np.hstack([b, h])
			meq = 1#A.shape[0]
		else:  # no equality constraint
			qp_C = -G.T
			qp_b = -h
			meq = 0
			
		return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
	
	def _nearestPD(self, A):
		"""
		Find the nearest positive-definite matrix to input
	
		A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
		credits [2].
	
		[1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
	
		[2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
		matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
		"""
	
		B = (A + A.T) / 2
		_, s, V = np.linalg.svd(B)
	
		H = np.dot(V.T, np.dot(np.diag(s), V))
	
		A2 = (B + H) / 2
	
		A3 = (A2 + A2.T) / 2
	
		if self._isPD(A3):
			return A3
	
		spacing = np.spacing(np.linalg.norm(A))
		# The above is different from [1]. It appears that MATLAB's `chol` Cholesky
		# decomposition will accept matrixes with exactly 0-eigenvalue, whereas
		# Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
		# for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
		# will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
		# the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
		# `spacing` will, for Gaussian random matrixes of small dimension, be on
		# othe order of 1e-16. In practice, both ways converge, as the unit test
		# below suggests.
		I = np.eye(A.shape[0])
		k = 1
		while not self._isPD(A3):
			mineig = np.min(np.real(np.linalg.eigvals(A3)))
			A3 += I * (-mineig * k**2 + spacing)
			k += 1
	
		return A3
	
	def _isPD(self, B):
		"""Returns true when input is positive-definite, via Cholesky"""
		try:
			_ = np.linalg.cholesky(B)
			return True
		except np.linalg.LinAlgError:
			return False