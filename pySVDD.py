import numpy as np
import quadprog

class SVDD:
	def __init__(self, C = 0.01, gamma = 3.):
		"""
		Initialize SVDD with Gaussian kernel function
		
		Parameters
		----------
		C : scalar
			SVDD allowable error constraint bound
		gamma : scalar
			Kernel function parameter
		"""
		self.C_ = C
		self.gamma_ = gamma
		
	def fit(self, X, y, sample_weight = None):
		"""
		Fit the model according to the given training data.
		
		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			Training vector, where n_samples in the number of samples and
			n_features is the number of features.
		y : array-like, shape = [n_samples]
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
		P, q, G, h, A, b = self._build_qp(X, y)
		beta = y*self._quadprog_solve_qp(self._nearestPD(P), q, G, h, A, b)
		self.support_vectors_ = X[abs(beta) > 1e-6,:]
		self.support_ = np.squeeze(np.where(abs(beta) > 1e-6))
		self.dual_coef_ = beta[abs(beta) > 1e-6]
		
		# Find decision threshold
		R2, _ = self._radius(self.support_vectors_)
		self.threshold_ = np.mean(R2)
		
		return self
	
	def predict(self, X):
		"""
		Predict data classification
		
		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			Test vector, where n_samples in the number of samples and
			n_features is the number of features.
			
		Returns
		-------
		y : array-like, shape = [n_samples]
			Target vector relative to X
		"""
		y = np.sign(self.decision_function(X)).astype('int')
		y[y == 0] = 1 # If data is on the threshold boundary, include it in the SVDD
		return y
	
	def decision_function(self, X):
		"""
		Test point radial difference from decision threshold.  Negative
		difference means test point is outside SVDD.
		
		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			Data vector, where n_samples in the number of samples and
			n_features is the number of features.
			
		Returns
		-------
		scalar shape = [n_samples]
			Radial difference from decision threshold
		"""
		radius, _ = self._radius(X)
		return self.threshold_ - radius
		
	def _radius(self, X):
		"""
		Test data hypersphere radii
		
		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			Data vector, where n_samples in the number of samples and
			n_features is the number of features.
			
		Returns
		-------
		R2 : array-like, shape = [n_samples]
			Hypersphere radius for each sample
		dR2 : array-like, shape = [n_samples]
			Hypersphere derivative evaluated at each sample
		"""
		kap = 0
		lam = 0
		mu = 0
		
		for i in range(len(self.dual_coef_)):
			Kxz, dKxz = self._rbf_kernel(self.support_vectors_[i,:], X)
			kap = kap + self.dual_coef_[i]*Kxz
			mu = mu + self.dual_coef_[i]*dKxz
			for j in range(len(self.dual_coef_)):
				Kxx, _ = self._rbf_kernel(self.support_vectors_[i,:], self.support_vectors_[j,:])
				lam = lam + self.dual_coef_[i]*self.dual_coef_[j]*Kxx
				
		R2 = 1 - 2*kap + lam
		dR2 = -2*mu
		
		return R2, dR2
		
	def _rbf_kernel(self, x, z):
		"""
		Gaussian (Radial Bias Function) kernel function
		
		Parameters
		----------
		x : array-like, shape = [n_features]
			Data point vector
		z : array-like, shape = [n_samples, n_features]
			Data vector, where n_samples in the number of samples and
			n_features is the number of features.
			
		Returns
		-------
		K : array-like, shape = [n_samples]
			Kernel for each sample
		dK2 : array-like, shape = [n_samples]
			Kernel derivative evaluated at each sample
		"""
		if z.ndim > 1:
			K = np.exp(-self.gamma_*np.linalg.norm(x - z, axis = 1)**2)
			#dK = 2*self.gamma*(x - z)*np.exp(-self.gamma*np.linalg.norm(x - z, axis = 1)**2)
			dK = np.dot(np.diag(np.exp(-self.gamma_*np.linalg.norm(x - z, axis = 1)**2)), 2*self.gamma_*(x - z))
		else:
			K = np.exp(-self.gamma_*np.linalg.norm(x - z)**2)
			dK = 2*self.gamma_*(x - z)*np.exp(-self.gamma_*np.linalg.norm(x - z)**2)
		
		return K, dK
		
	def _build_qp(self, X, y):
		"""
		Construct quadratic programming elements
		
		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			Data vector, where n_samples in the number of samples and
			n_features is the number of features.
		y : array-like, shape = [n_samples]
			Target vector relative to X
			
		Returns
		-------
		P : array-like, shape = [n_samples, n_samples]
		q : array-like, shape = [n_samples]
		G : array-like, shape = [n_constraints, n_samples]
		h : array-like, shape = [n_constraints]
		A : array-like, shape = [n_constraints, n_samples]
		b : array-like, shape = [n_constraints]
		"""
		# Build P
		P = np.eye(len(y))
		for i in range(len(y) - 1):
			for j in range(i + 1, len(y)):
				Kxx, _ = self._rbf_kernel(X[i,:],X[j,:])
				P[i,j] = y[i]*y[j]*Kxx
		P = self._nearestPD(P + P.T - np.eye(len(y)))
		
		# Build q
		q = np.zeros(len(y))
		
		# Build G
		G = np.vstack((-np.eye(len(y)),np.eye(len(y))))
		
		# Build h
		h = np.hstack((np.zeros(len(y)),self.C_*np.ones(len(y))))
		
		# Build A
		A = y
		
		# Build b
		b = 1.
		
		return P, q, G, h, A, b
	
	def _quadprog_solve_qp(self, P, q, G = None, h = None, A = None, b = None):
		"""
		Quadratic programming solver interface.  Solves the following:
		
		min		0.5*x.T*P*x + q.T*x
		s.t.	G*x <= h
				A*x = b
				
		This is CVXOPT notation but we use quadprog.  This method translates
		to quadprog notation.
		
		Parameters
		----------
		P : array-like, shape = [n_samples, n_samples]
		q : array-like, shape = [n_samples]
		G : array-like, shape = [n_constraints, n_samples]
		h : array-like, shape = [n_constraints]
		A : array-like, shape = [n_constraints, n_samples]
		b : array-like, shape = [n_constraints]
			
		Returns
		-------
		array-like, shape = [n_samples]
			QP solution x
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