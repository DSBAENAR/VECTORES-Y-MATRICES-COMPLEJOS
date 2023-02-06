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
    return [list(x) for x in zip(*m)]

def conjugate_matrix(m):
    return np.conjugate(m)

def adjoint_matrix(m):
    return np.conjugate(transpose_matrix(m))

def matrix_multiplication(m1, m2):
    if len(m1[0]) != len(m2):
        return 'No se puede realizar la multilplicacion'
    return np.dot(m1, m2)
    

def matrix_vector_multiplication(m, v):
    return np.dot(m, v)

def inner_product_complex_vectors(v1, v2):
    return np.dot(conjugate_matrix(v1), v2)

def norm_complex_vector(v):
    return np.sqrt(np.dot(conjugate_matrix(v), v))

def distance_complex_vectors(v1, v2):
    return norm_complex_vector(v1 - v2)

def is_unitary_matrix(m):
    return np.allclose(np.dot(m, np.conjugate(np.transpose(m))), np.identity(m.shape[0]))

def is_hermitian_matrix(m):
    return np.allclose(m, np.conjugate(np.transpose(m)))

def tensor_product(m1, m2):
    return np.kron(m1, m2)
