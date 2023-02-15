import numpy as np
import unittest
import ComplexMatrix

class TestQuantumOperations(unittest.TestCase):
    def setUp(self):
        self.complex_vector1 = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        self.complex_vector2 = np.array([1 + 1j, 2 + 2j, 4 + 4j])
        self.complex_matrix1 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        self.complex_matrix2 = np.array([[2 + 2j, 1 + 1j], [1 + 1j, 2 + 2j]])

    def test_add_complex_vectors_1(self):
        result = ComplexMatrix.add_complex_vectors(self.complex_vector1, self.complex_vector2)
        expected = np.array([2 + 2j, 4 + 4j, 7 + 7j])
        np.testing.assert_array_equal(result, expected)

    def test_add_complex_vectors_2(self):
        result = ComplexMatrix.add_complex_vectors(np.array([7 + 3j, 9 + 5j, 2 + 4j]),np.array([1 + 16j, 23 + 8j, 11 + 46j]))
        expected = np.array([8+ 19j, 32 + 13j, 13 + 50j])
        np.testing.assert_array_equal(result, expected)
        
    def test_addition_with_zero_vector(self):
        vector1 = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        vector2 = np.zeros(3, dtype=complex)
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        result = ComplexMatrix.add_complex_vectors(vector1, vector2)
        np.testing.assert_array_equal(result, expected)

    def sub_complex_vectors_1(self):
        vector1 = np.array([1+2j, 3-1j])
        vector2 = np.array([2+1j, 4+0j])
        result=ComplexMatrix.sub_complex_vectors(vector1,vector2)
        expected=np.array([-1+1j,-1-1j])
        np.testing.assert_equal(result,expected)
    
    def test_sub_complex_vectors_2(self):
        result=ComplexMatrix.sub_complex_vectors(self.complex_vector1,self.complex_vector2)
        expected=np.array([0,0,-1-1j])
        np.testing.assert_equal(result,expected)

    def test_scalar_multiplication_complex_vector_1(self):
        result = ComplexMatrix.scalar_multiplication_complex_vector(2, self.complex_vector1)
        expected = np.array([2 + 2j, 4 + 4j, 6 + 6j])
        np.testing.assert_array_equal(result, expected)
    
    def test_scalar_multiplication_complex_vector_2(self):
        result = ComplexMatrix.scalar_multiplication_complex_vector(10, self.complex_vector1)
        expected = np.array([10+ 10j, 20 + 20j, 30 + 30j])
        np.testing.assert_array_equal(result, expected)

    def test_add_complex_matrices_1(self):
        result = ComplexMatrix.add_complex_matrices(self.complex_matrix1, self.complex_matrix2)
        expected = np.array([[3 + 3j, 3 + 3j], [4 + 4j, 6 + 6j]])
        np.testing.assert_array_equal(result, expected)

    def test_add_complex_matrices_1(self):
        A=np.array([[6+1j,7+2j],[1-5j,4-3j]])
        B=np.array([[-9+1j,7-10j],[3-1j,-3+5j]])
        result = ComplexMatrix.add_complex_matrices(A,B)
        expected = np.array([[-3 + 2j, 14 -8j], [4 -6j, 1 + 2j]])
        np.testing.assert_array_equal(result, expected)


    def test_sub_complex_matrices_1(self):
        result = ComplexMatrix.inverse_complex_matrix(self.complex_matrix1,self.complex_matrix2)
        expected = np.array([[-1-1j,1+1j],[2+2j,2+2j]])
        np.testing.assert_array_equal(result, expected)

    def test_sub_complex_matrices_2(self):
        A=np.array([[6+1j,7+2j],[1-5j,4-3j]])
        B=np.array([[-9+1j,7-10j],[3-1j,-3+5j]])
        result = ComplexMatrix.inverse_complex_matrix(A,B)
        expected = np.array([[15,12j],[-2-4j,7-8j]])
        np.testing.assert_array_equal(result, expected)

    def test_scalar_matrix_product_1(self):
        c = 2+3j
        M = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
        expected_output = np.array([[-4+7j, -6+17j], [-8+27j,-10+37j]])
        self.assertTrue((ComplexMatrix.scalar_multiplication_complex_matrix(c, M) == expected_output).all())
    
    def test_scalar_matrix_product_2(self):
        c = 5+2j
        M = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
        expected_output = np.array([[1+12j, 7+26j], [13+40j,19+54j]])
        self.assertTrue((ComplexMatrix.scalar_multiplication_complex_matrix(c, M) == expected_output).all())

    def test_matrix_transpose_1(self):
        A = np.array([[2, 1, 3, -2], [5, -3, 1, 1]])
        B = np.array([[2,5],[1,-3],[3,1],[-2,1]])
        np.testing.assert_equal(ComplexMatrix.transpose_matrix(A), B)
    
    def test_matrix_transpose_2(self):
        result = ComplexMatrix.transpose_matrix(self.complex_matrix1) 
        expected = ([[1+1j,3+3j],[2+2j,4+4j]])
        self.assertEqual(result, expected)

    def test_conjugate_matrix_1(self):
        result = ComplexMatrix.conjugate_matrix(self.complex_matrix1)
        expected = np.array([[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j]])
        np.testing.assert_array_equal(result, expected)
    
    def test_conjugate_matrix_2(self):
        result = ComplexMatrix.conjugate_matrix(self.complex_matrix2)
        expected=np.array([[2-2j,1-1j],[1-1j,2-2j]])
        np.testing.assert_array_equal(result, expected)

    def test_adjoint_matrix_1(self):
        result = ComplexMatrix.adjoint_matrix(self.complex_matrix1)
        expected = np.array([[1-1j,3-3j],[2-2j,4-4j]])
        np.testing.assert_array_equal(result,expected)

    def test_adjoint_matrix_2(self):
        result = ComplexMatrix.adjoint_matrix(self.complex_matrix2)
        expected = np.array([[2-2j,1-1j],[1-1j,2-2j]])
        np.testing.assert_array_equal(result,expected)

    def test_matrix_multiplication_1(self):
        A = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        result = ComplexMatrix.matrix_multiplication(self.complex_matrix1, self.complex_matrix2)
        expected = np.array([[0+8j,0+10j], [0+20j, 0+22j]])
        np.testing.assert_array_equal(result, expected)

    def test_matrix_multiplication_2(self):
        A = np.array([[2, 1, 3, -2], [5, -3, 1, 1]])
        B = np.array([[2,5],[1,-3],[3,1],[-2,1]])
        result = ComplexMatrix.matrix_multiplication(A,B)
        expected = np.array([[18,8], [8,36]])
        np.testing.assert_array_equal(result, expected)


    def test_matrix_vector_multiplication_1(self):
        A=np.array([[2 + 2j, 1 + 1j], [1 + 1j, 2 + 2j]])
        B=np.array([1 + 1j, 2 + 2j])
        result = ComplexMatrix.matrix_vector_multiplication(A, B)
        expected = np.array([(8j),(10j)])
        np.testing.assert_equal(result,expected)

    def test_matrix_vector_multiplication_2(self):
        A = np.array([[2 +1j, 3-2j], [5-3j, 1-1j]])
        B = np.array([1+2j, 3-1j])
        result = ComplexMatrix.matrix_vector_multiplication(A,B)
        expected = np.array([7-4j,13+3j])
        np.testing.assert_equal(result,expected)

    def test_inner_product_1(self):
        result = ComplexMatrix.inner_product_complex_vectors(self.complex_vector1, self.complex_vector2)
        expected = 34
        self.assertEqual(result, expected)

    def test_vector_norm_1(self):
        result = ComplexMatrix.norm_complex_vector(self.complex_vector1)
        expected = np.sqrt(28)
        self.assertAlmostEqual(result, expected, places=9)
    
    def test_vector_norm_2(self):
        result = ComplexMatrix.norm_complex_vector(self.complex_vector2)
        expected = np.sqrt(42)
        self.assertAlmostEqual(result, expected, places=9)

    def test_distance_between_vectors_1(self):
        result = ComplexMatrix.distance_complex_vectors(self.complex_vector1, self.complex_vector2)
        expected = np.sqrt(2)
        self.assertAlmostEqual(result, expected, places=9)
    
    def test_distance_between_vectors_2(self):
        A = np.array([1 + 2j, 3 + 4j])
        B = np.array([1+2j, 3-1j])
        result = ComplexMatrix.distance_complex_vectors(A,B)
        expected = 5
        self.assertAlmostEqual(result, expected, places=9)

    def test_unitary_matrix_1(self):
        result = ComplexMatrix.is_unitary_matrix(self.complex_matrix1)
        self.assertFalse(result)

    def test_unitary_matrix_2(self):
        A = np.array([[(1j+1)/2,1j/np.sqrt(3),(3+1j)/(2*np.sqrt(15))],[-1/2,1/(np.sqrt(3)),(4+3j)/(2*np.sqrt(15))],[1/2,-1j/(np.sqrt(3)),(5j)/(2*np.sqrt(15))]])
        result = ComplexMatrix.is_unitary_matrix(A)
        expected = True
        np.testing.assert_equal(result,expected)

    def test_hermitian_matrix_1(self):
        result = ComplexMatrix.is_hermitian_matrix(self.complex_matrix1)
        self.assertFalse(result)

    def test_hermitian_matrix_1(self):
        result = ComplexMatrix.is_hermitian_matrix(self.complex_matrix1)
        self.assertFalse(result)

    def test_hermitian_matrix_2(self):
        A = np.array([[5 , 4 + 5j , 6 - 16j],[4 - 5j , 13 , 7],[6 + 16j , 7 , -2.1]])
        result = ComplexMatrix.is_hermitian_matrix(A)
        expected = True
        np.testing.assert_equal(result,expected)

    def test_tensor_product_1(self):
        result = ComplexMatrix.tensor_product(self.complex_vector1, self.complex_vector2)
        expected = np.array([(2j),(4j),(8j),(4j),(8j),(16j),(6j),(12j),(24j)])
        np.testing.assert_array_equal(result, expected)
        
    def test_tensor_product_2(self):
        A = np.array([1+2j, 3-1j])
        B = np.array([2+1j, 4+0j])
        result = ComplexMatrix.tensor_product(A,B)
        expected = np.array([(5j),(4 + 8j),(7 + 1j),(12 - 4j)])
        np.testing.assert_array_equal(result, expected)

    def test_eigen_values_1(self):
        result=ComplexMatrix.Eigen_Values(self.complex_matrix1)
        expected=np.array([-0.37228132-0.37228132j, 5.37228132+5.37228132j])
        np.testing.assert_array_almost_equal(result,expected)
    
    def test_eigen_values_2(self):
        result=ComplexMatrix.Eigen_Values(self.complex_matrix2)
        expected=np.array([3 + 3j, 1 + 1j])
        np.testing.assert_array_almost_equal(result,expected)
    
    
    def test_eigen_vectors_1(self):
        result=ComplexMatrix.Eigen_Vectors(self.complex_matrix1)
        expected=np.array([[0.82456484+0.00000000e+00j,0.41597356-8.32667268e-17j],[-0.56576746+1.11022302e-16j,0.90937671+0.00000000e+00j]])
        np.testing.assert_array_almost_equal(result,expected)
    
    def test_eigen_vectors_2(self):
        result=ComplexMatrix.Eigen_Vectors(self.complex_matrix2)
        expected=np.array([[0.70710678+0.00000000e+00j,-0.70710678+3.33066907e-16j],[0.70710678+1.11022302e-16j,0.70710678+0.00000000e+00j]])
        np.testing.assert_array_almost_equal(result,expected)


    

if __name__ == '_main_':
    unittest.main()


    def test_tensor_product(self):
        result = ComplexMatrix.tensor_product(self.complex_vector1, self.complex_vector2)
        expected = np.array([(2j),(4j),(8j),(4j),(8j),(16j),(6j),(12j),(24j)])
        np.testing.assert_array_equal(result, expected)

if __name__ == '_main_':
    unittest.main()
