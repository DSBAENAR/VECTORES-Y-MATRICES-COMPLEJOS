import numpy as np
import unittest
import ComplexMatrix

class TestQuantumOperations(unittest.TestCase):
    def setUp(self):
        self.complex_vector1 = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        self.complex_vector2 = np.array([1 + 1j, 2 + 2j, 4 + 4j])
        self.complex_matrix1 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        self.complex_matrix2 = np.array([[2 + 2j, 1 + 1j], [1 + 1j, 2 + 2j]])

    def test_add_complex_vectors(self):
        result = ComplexMatrix.add_complex_vectors(self.complex_vector1, self.complex_vector2)
        expected = np.array([2 + 2j, 4 + 4j, 7 + 7j])
        np.testing.assert_array_equal(result, expected)
        
    def test_addition_with_zero_vector(self):
        vector1 = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        vector2 = np.zeros(3, dtype=complex)
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        result = ComplexMatrix.add_complex_vectors(vector1, vector2)
        np.testing.assert_array_equal(result, expected)

    def test_inverse_complex_vector(self):
        result = ComplexMatrix.inverse_complex_vector(self.complex_vector1)
        expected = np.array([-1 - 1j, -2 - 2j, -3 - 3j])
        np.testing.assert_array_equal(result, expected)

    def test_scalar_multiplication_complex_vector(self):
        result = ComplexMatrix.scalar_multiplication_complex_vector(2, self.complex_vector1)
        expected = np.array([2 + 2j, 4 + 4j, 6 + 6j])
        np.testing.assert_array_equal(result, expected)

    def test_add_complex_matrices(self):
        result = ComplexMatrix.add_complex_matrices(self.complex_matrix1, self.complex_matrix2)
        expected = np.array([[3 + 3j, 3 + 3j], [4 + 4j, 6 + 6j]])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_complex_matrix(self):
        result = ComplexMatrix.inverse_complex_matrix(self.complex_matrix1)
        expected = np.array([[-1 - 1j, -2 - 2j], [-3 - 3j, -4 - 4j]])
        np.testing.assert_array_equal(result, expected)

    def test_scalar_matrix_product(self):
        c = complex(2, 3)
        M = np.array([[complex(1, 2), complex(3, 4)], [complex(5, 6), complex(7, 8)]])
        expected_output = np.array([[complex(-4, 7), complex(-6, 17)], [complex(-8, 27), complex(-10, 37)]])
        self.assertTrue((ComplexMatrix.scalar_multiplication_complex_matrix(c, M) == expected_output).all())

    def test_matrix_transpose(self):
        A = ([[(2, 1), (3, -2)], [(5, -3), (1, 1)]])
        B = ([[(2, 3, 5, 1)], [(1, -2, -3, 1)]])
        self.assertTrue(ComplexMatrix.transpose_matrix(A), B)

        
    def test_conjugate_matrix(self):
        result = ComplexMatrix.conjugate_matrix(self.complex_matrix1)
        expected = np.array([[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j]])
        np.testing.assert_array_equal(result, expected)

    def test_adjoint_matrix(self):
        result = ComplexMatrix.adjoint_matrix(np.array([[5,1 + 3j, 4 + 5j]]))
        expected = np.array([[(5)],[1-3j],[4-5j]])
        np.testing.assert_array_equal(result,expected)
        

    def test_matrix_multiplication(self):
        result = ComplexMatrix.matrix_multiplication(self.complex_matrix1, self.complex_matrix2)
        expected = np.array([[8j,10j], [20j, 22j]])
        np.testing.assert_array_equal(result, expected)

        

    def test_matrix_vector_action(self):
        A = np.array([[(5), (1+ 3j)]])
        result = ComplexMatrix.matrix_vector_multiplication(A, self.complex_vector1)
        expected = np.array([[(5+5j),(10+10j),(15+15j)],[(-2+4j),(-4+8j),(-6+12j)]])
        np.testing.assert_array_equal(result, expected)

    def test_inner_product(self):
        result = ComplexMatrix.inner_product_complex_vectors(self.complex_vector1, self.complex_vector2)
        expected = 34
        self.assertEqual(result, expected)

    def test_vector_norm(self):
        result = ComplexMatrix.norm_complex_vector(self.complex_vector1)
        expected = np.sqrt(28)
        self.assertAlmostEqual(result, expected, places=9)

    def test_distance_between_vectors(self):
        result = ComplexMatrix.distance_complex_vectors(self.complex_vector1, self.complex_vector2)
        expected = np.sqrt(2)
        self.assertAlmostEqual(result, expected, places=9)

    def test_unitary_matrix(self):
        result = ComplexMatrix.is_unitary_matrix(self.complex_matrix1)
        self.assertFalse(result)

    def test_hermitian_matrix(self):
        result = ComplexMatrix.is_hermitian_matrix(self.complex_matrix1)
        self.assertFalse(result)

    def test_tensor_product(self):
        result = ComplexMatrix.tensor_product(self.complex_vector1, self.complex_vector2)
        expected = np.array([(2j),(4j),(8j),(4j),(8j),(16j),(6j),(12j),(24j)])
        np.testing.assert_array_equal(result, expected)

if __name__ == '_main_':
    unittest.main()
