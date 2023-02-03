import numpy as np

def add_complex_vectors(v1, v2):
    return np.add(v1, v2)

def inverse_complex_vector(v):
    return np.negative(v)

def scalar_multiplication_complex_vector(c, v):
    return np.multiply(c, v)

def add_complex_matrices(m1, m2):
    return np.add(m1, m2)

def inverse_complex_matrix(m):
    return np.negative(m)

def scalar_multiplication_complex_matrix(c, m):
    return np.multiply(c,m)

def transpose_matrix(m):
    return np.transpose(m)

def conjugate_matrix(m):
    return np.conjugate(m)

def adjoint_matrix(m):
    return np.conjugate(np.transpose(m))

def matrix_multiplication(m1, m2):
    return np.dot(m1, m2)

def matrix_vector_multiplication(m, v):
    return np.dot(m, v)
def inner_product_complex_vectors(v1, v2):
    return np.dot(np.conjugate(v1), v2)

def norm_complex_vector(v):
    return np.sqrt(np.dot(np.conjugate(v), v))

def distance_complex_vectors(v1, v2):
    return norm_complex_vector(v1 - v2)

def is_unitary_matrix(m):
    return np.allclose(np.dot(m, np.conjugate(np.transpose(m))), np.identity(m.shape[0]))

def is_hermitian_matrix(m):
    return np.allclose(m, np.conjugate(np.transpose(m)))

def tensor_product(m1, m2):
    return np.kron(m1, m2)
