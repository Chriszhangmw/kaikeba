


from scipy.sparse import csr_matrix
import numpy as np

indptr = np.array([0,2,3,6])
indices = np.array([0,2,2,0,1,2])
data = np.array([1,2,3,4,5,6])
csr_matrix_0 = csr_matrix((data,indices,indptr),shape=(3,3))
print(csr_matrix_0.toarray())

csr_matrix_0 = csr_matrix_0 + csr_matrix_0.T.multiply(csr_matrix_0.T > csr_matrix_0) - csr_matrix_0.multiply(csr_matrix_0.T > csr_matrix_0)
print(csr_matrix_0.toarray())







