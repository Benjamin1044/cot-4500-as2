import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def nevilles_method(x_points, y_points, x):
  
    size = len(y_points)
    matrix = np.zeros((size,size))
    
   
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
    
    
    num_of_points = len(x_points)
    
    
    for i in range(1, num_of_points):
        for j in range(1,(i+1)):
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]

            denominator = x_points[i] - x_points[i-j]

            
            coefficient = (first_multiplication - second_multiplication)/denominator
            matrix[i][j] = coefficient

    
    return matrix[num_of_points - 1][num_of_points - 1]












def divided_difference_table(x_points, y_points):
    
    size = len(y_points)
    matrix = np.zeros((size,size))

   
    for index, row in enumerate(matrix):
        row[0] = y_points[index]
    
    
    
    for i in range(1, size):
        for j in range(1, (i+1)):
            
            numerator = (matrix[i][j-1] - matrix[i-1][j-1])

            
            denominator = (x_points[i] - x_points[i - j])

            operation = numerator / denominator

            
            matrix[i][j] = '{0:.7g}'.format(operation)


    print( matrix[1][1] )
    print( matrix[2][2] ) 
    print( matrix[3][3] )
    return matrix


  
def get_approximate_result(matrix, x_points, value):
    
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    size = len(matrix)
    
    for index in range(1, size):
        
        polynomial_coefficient = matrix[index][index]

        
        reoccuring_x_span *= (value - x_points[index-1])
        
        
        mult_operation = polynomial_coefficient * reoccuring_x_span

        
        reoccuring_px_result += mult_operation

    
    
    return reoccuring_px_result



def apply_div_dif(matrix):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
           
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            
            left = matrix[i][j-1]

            
            diagonal_left = matrix[i-1][j-1]

            
            numerator = (left - diagonal_left)

            
            denominator = matrix[i][0] - matrix[i - j + 1][0]

            
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix


def hermite_interpolation():
    num_of_pointsnum_of_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]

     
    num_of_points = len(num_of_pointsnum_of_points)
    doublepoint = 2 * num_of_points
    matrix = np.zeros((doublepoint,doublepoint))

    
    for x in range(num_of_points):
      matrix[2*x][0] = num_of_pointsnum_of_points[x]
      matrix[2*x+1][0] = num_of_pointsnum_of_points[x]
        
    
   
    for x in range(num_of_points):
      matrix[2*x][1] = y_points[x]
      matrix[2*x+1][1] = y_points[x]

    
    for x in range(num_of_points):
      matrix[2*x+1][2] = slopes[x]
        

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)

def naturalcubicspline_matrix(x_points, fx_points):
  n = len(x_points)
  c = np.zeros(n-1)
  h = x_points

  for i in range(n-1):
    h[i] = x_points[i+1] - x_points[i] 
    
  for i in range(n-1):
    c[i] = (fx_points[i+1] - fx_points[i]) / h[i]
    
    
  A = np.zeros((n, n))
  b = np.zeros(n)
  A[0][0] = 1
  A[n-1][n-1] = 1
  for i in range(1, n-1):
    A[i][i-1] = h[i-1]
    A[i][i] = 2 * (h[i] + h[i-1])
    A[i][i+1] = h[i]
    
    b[i] = 3 * ( c[i] - c[i-1]) 
  
  x = np.linalg.solve(A, b)
  print("")
  print(A)
  print("")
  print(b)
  print("")
  print(x)
if __name__ == "__main__":
    x_points1 = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7

    print(nevilles_method(x_points1, y_points, approximating_value))
    print("")

    x_points2 = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    matrix = divided_difference_table(x_points2, y_points)
    print("")

    value = 7.3
    final_approximation = get_approximate_result(matrix, x_points2, value)
    print(final_approximation)
    print("")

    hermite_interpolation()
    print("")
    x_points = [2, 5, 8, 10]
    fx_points = [3, 5, 7, 9]
    
    naturalcubicspline_matrix(x_points, fx_points)