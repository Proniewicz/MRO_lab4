from sklearn.linear_model import Lasso
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
from matplotlib.pyplot import plot, show, figure, title
import numpy as np

N = 5000
FS = 4e4
M = 500
f1, f2 = 697, 1336 # Pick any two touchtone frequencies
duration = 1./8
t = np.linspace(0, duration, duration*FS)
f = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
f = np.reshape(f, (len(f),1))

# Displaying the test signal
plot(t,f)
title('Original Signal')
show()

# Randomly sampling the test signal
k = np.random.randint(0,N,(M,))
k = np.sort(k) # making sure the random samples are monotonic
b = f[k]
plot(t,f,'b', t[k],b,'r.')
title('Original Signal with Random Samples')
show()

D = dct(np.eye(N))
A = D[k,:]

lasso = Lasso(alpha=0.001)
lasso.fit(A,b.reshape((M,)))

# Plotting the reconstructed coefficients and the signal
plot(lasso.coef_)
title('IDCT of the Reconstructed Signal')
recons = dct(lasso.coef_.reshape((N,1)),axis=0)
figure()
plot(t,recons)
title('Reconstucted Signal')
show()

recons_sparse = coo_matrix(lasso.coef_)
sparsity = 1 - float(recons_sparse.getnnz())/len(lasso.coef_)
print (sparsity)
