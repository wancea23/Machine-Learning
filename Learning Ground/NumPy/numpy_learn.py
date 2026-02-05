import numpy as np

#1
# # print(np.__version__)

# my_list = [1,2,3,4,5]

# array = np.array(my_list)
# array = array / 5

# print(my_list)
# print(array)

#2
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])

# print(a[0]) #first row
# print(a[0][0]) #first row first column
# print(a.shape) #shape of array n row n column
# print(a.ndim) #number of dimensions

#3
# a = np.array([[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12],
#               [13,14,15,16]])
# #a[start:end:step] end is exclusive
# print(a)
# print("\n")
# print(a[0:3,0:2])
# print("\n")
# print(a[0:4:2,0:2])
# print("\n")
# print(a[:, 2])

#4
# a = np.array([[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12],
#               [13,14,15.55,16]])

# a2 = np.array([[1,2,3,4],
#               [5,6,7,8],
#                 [9,10,11,12],
#                 [13,14,15.55,16]])

# print(a + 1) #element wise addition
# print("\n")
# print(a - 4) #element wise subtraction
# print("\n")
# print(a / 2) #element wise division
# print("\n")
# print(a * 2) #element wise multiplication
# print("\n")
# print(a ** 5) #element wise exponentiation
# print("\n")
# print(np.sqrt(a)) #square root
# print("\n")
# print(np.round(a)) #round to nearest integer
# print("\n")
# print(np.floor(a)) #round down
# print("\n")
# print(np.ceil(a)) #round up
# print("\n")
# print(a * np.pi) #element wise multiplication
# print("\n")
# print(a ** a2) #element wise exponentiation
# print("\n")
# print(np.dot(a,a2)) #matrix multiplication
# print("\n")
# print(a @ a2) #matrix multiplication
# print("\n")
# print(a>1) #boolean array
# print("\n")
# print(a[a>1]) #filtering array
# print("\n")
# print(a[a>1] * 2) #filtering and element wise multiplication
# print("\n")
# print(np.sum(a)) #sum of all elements
# print("\n")
# print(np.sum(a, axis=1)) #sum of each column 0 axis for row axis 1 for column
# print("\n")
# a[a>1] = 69 #boolean array assignment
# print(a)

#5 Broadcasting gotta match either n of rows or n of columns or be 1
# a1 = np.array([[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12],
#               [13,14,15.55,16]])

# a2 = np.array([[1],
#                [2],
#                [3],
#                [4]])

# print(a1.shape)
# print(a2.shape)

# print(a1 + a2) #broadcasting addition
# print("\n")
# print(a1 ** a2) #broadcasting exponentiation

#6 Aggregation Functions
# a = np.array([[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12],
#               [13,14,15,16]])

# print("Minimum value in the array:")
# print(np.min(a)) #minimum value
# print("\n")
# print("Maximum value in the array:")
# print(np.max(a)) #maximum value
# print("\n")
# print("Mean value of the array:")
# print(np.mean(a)) #mean value
# print("\n")
# print("Median value of the array:")
# print(np.median(a)) #median value
# print("\n")
# print("Standard deviation of the array:")
# print(np.std(a)) #standard deviation
# print("\n")
# print("Variance of the array:")
# print(np.var(a)) #variance
# print("\n")
# print("Index of minimum value in the array:")
# print(np.argmin(a)) #index of minimum value
# print("\n")
# print("Index of maximum value in the array:")
# print(np.argmax(a)) #index of maximum value
# print("\n")
# print("Sum of all elements in the array:")
# print(np.sum(a)) #sum of all elements
# print("\n")
# print("50th percentile of the array:")
# print(np.percentile(a, 50)) #50th percentile (median)
# print("\n")
# print("25th percentile of the array:")
# print(np.percentile(a, 25)) #25th percentile
# print("\n")
# print("75th percentile of the array:")
# print(np.percentile(a, 75)) #75th percentile
# print("\n")
# print("Sorted array column:")
# print(np.sort(a, axis=0)) #sort each column
# print("\n")
# print("Sorted array row:")
# print(np.sort(a, axis=1)) #sort each row
# print("\n")
# print("Unique values in the array:")
# print(np.unique(a)) #unique values in array
# print("\n")
# print("Transpose of the array:")
# print(np.transpose(a)) #transpose of array
# print("\n")
# print("Transpose of the array using .T:")
# print(a.T) #transpose of array
# print("\n")
# print("Clip values in the array below 5 to 5 and above 10 to 10:")
# print(np.clip(a, 5, 10)) #clip values below 5 to 5 and above 10 to 10
# print("\n")
# print("Conditional selection (High if >10 else Low):")
# print(np.where(a > 10, 'High', 'Low')) #conditional selection
# print("\n")
# print("Check if any value is greater than 15:")
# print(np.any(a > 15)) #check if any value is greater than 15
# print("\n")
# print("Check if all values are less than 20:")
# print(np.all(a < 20)) #check if all values are less than 20
# print("\n")
# print("Cumulative sum of the array:")
# print(np.cumsum(a)) #cumulative sum
# print("\n")
# print("Cumulative product of the array:")
# print(np.cumprod(a)) #cumulative product
# print("\n")
# print("Difference between consecutive rows:")
# print(np.diff(a, axis=0)) #difference between consecutive rows
# print("\n")
# print("Difference between consecutive columns:")
# print(np.diff(a, axis=1)) #difference between consecutive columns
# print("\n")
# print("Meshgrid of two arrays:")
# print(np.meshgrid(np.array([1,2,3]), np.array([4,5,6]))) #create coordinate matrices
# print("\n")
# print("Flattened array:")
# print(np.ravel(a)) #flatten the array
# print("\n")
# print("Reshaped array to 8 rows and 2 columns:")
# print(np.reshape(a, (8,2))) #reshape array to 8 rows and 2 columns
# print("\n")
# print("Flattened array using .flatten():")
# print(np.hstack((a, a))) #horizontal stack
# print("\n")
# print("Stacked array along new third axis:")
# print(np.vstack((a, a))) #vertical stack
# print("\n")
# print("Stacked array along depth (third axis):")
# print(np.dstack((a, a))) #depth stack
# print("\n")
# print("Split array into 2 along rows:")
# print(np.split(a, 2, axis=0)) #split array into 2 along rows
# print("\n")
# print("Split array into 2 along columns:")
# print(np.split(a, 2, axis=1)) #split array into 2 along columns
# print("\n")
# print("Tiled array:")
# print(np.tile(a, (2,2))) #tile array
# print("\n")
# print("Repeated rows twice:")
# print(np.repeat(a, 2, axis=0)) #repeat each row twice
# print("\n")
# print("Repeated columns twice:")
# print(np.repeat(a, 2, axis=1)) #repeat each column twice
# print("\n")
# print("Diagonal of the array:")
# print(np.fill_diagonal(a, 0)) #fill diagonal with 0
# print(a)
# print("\n")
# print("Trace of the array:")
# print(np.trace(a)) #sum of diagonal elements
# print("\n")
# print("Determinant of the array:")
# print(np.linalg.det(a)) #determinant of matrix
# print("\n")
# print("Inverse of the array:")
# print(np.linalg.inv(a)) #inverse of matrix
# print("\n")
# print("Eigenvalues and eigenvectors of the array:")
# print(np.linalg.eig(a)) #eigenvalues and eigenvectors
# print("\n")
# print("Singular value decomposition of the array:")
# print(np.linalg.svd(a)) #singular value decomposition
# print("\n")
# print("Rank of the array:")
# print(np.linalg.matrix_rank(a)) #rank of matrix
# print("\n")
# print("Solve linear equations Ax = b where b is a vector of ones:")
# print(np.linalg.solve(a, np.ones((4,)))) #solve linear equations Ax = b where b is a vector of ones
# print("\n")
# print("Least squares solution to Ax = b where b is a vector of ones:")
# print(np.linalg.lstsq(a, np.ones((4,)), rcond=None)) #least squares solution to Ax = b where b is a vector of ones
# print("\n")
# print("Frobenius norm of the array:")
# print(np.linalg.norm(a)) #Frobenius norm of matrix
# print("\n")
# print("Condition number of the array:")
# print(np.linalg.cond(a)) #condition number of matrix
# print("\n")
# print("QR decomposition of the array:")
# print(np.linalg.qr(a)) #QR decomposition
# print("\n")
# print("LU decomposition of the array:")
# print(np.linalg.slogdet(a + 16 * np.eye(4))) #sign and log determinant
# print("\n")
# print("Pseudo-inverse of the array:")
# print(np.linalg.pinv(a)) #pseudo-inverse of matrix
# print("\n")
# print("Tensor inverse of reshaped array:")
# print(np.linalg.tensorinv(a.reshape(2,2,4))) #tensor inverse
# print("\n")
# print("Tensor solve of reshaped array with ones:")
# print(np.linalg.tensorsolve(a.reshape(2,2,4), np.ones((2,2)))) #tensor solve
# print("\n")
# print("Multiple matrix dot product of a, a.T, and a:")
# print(np.linalg.multi_dot([a, a.T, a])) #multiple matrix dot product

#7 Random Number Generation
# print("Random float between 0 and 1:")
# print(np.random.random())
# print("\n")
# print("Random integer between 0 and 10:")
# print(np.random.randint(0, 10))
# print("\n")
# print("Random array of 5 floats between 0 and 1:")
# print(np.random.rand(5))
# print("\n")
# print("Random array of 3 integers between 0 and 10:")
# print(np.random.randint(0, 10, size=3))
# print("\n")
# print("Random array of shape (2,3) with floats between 0 and 1:")
# print(np.random.rand(2,3))
# print("\n")
# print("Random array of shape (2,3) with integers between 0 and 10:")
# print(np.random.randint(0, 10, size=(2,3)))

#8 Random Sampling
# print("Random sample from a 1D array:")
# print(np.random.choice(np.array([1,2,3,4,5]), size=3, replace=False))
# print("\n")
# print("Random sample from a 2D array:")
# print(np.random.choice(np.array([[1,2],[3,4],[5,6]]).flatten(), size=4, replace=False))
# print("\n")

#9 Random Permutations
# print("Random permutation of a 1D array:")
# print(np.random.permutation(np.array([1,2,3,4,5])))
# print("\n")
# print("Random permutation of a 2D array:")
# print(np.random.permutation(np.array([[1,2],[3,4],[5,6]])))

#10 Random Seed
# np.random.seed(42)
# print("Random float between 0 and 1 with seed 42:")
# print(np.random.random())
# print("\n")
# print("Random integer between 0 and 10 with seed 42:")
# print(np.random.randint(0, 10))
# print("\n")
# print("Random array of 5 floats between 0 and 1 with seed 42:")
# print(np.random.rand(5))
# print("\n")
# print("Random array of 3 integers between 0 and 10 with seed 42:")
# print(np.random.randint(0, 10, size=3))
# print("\n")
# print("Random array of shape (2,3) with floats between 0 and 1 with seed 42:")
# print(np.random.rand(2,3))
# print("\n")
# print("Random array of shape (2,3) with integers between 0 and 10 with seed 42:")
# print(np.random.randint(0, 10, size=(2,3)))

#11 Random Distributions
# print("Random sample from a normal distribution with mean 0 and std 1:")
# print(np.random.normal(0, 1))
# print("\n")
# print("Random sample from a normal distribution with mean 5 and std 2:")
# print(np.random.normal(5, 2))
# print("\n")
# print("Random array of 5 samples from a normal distribution with mean 0 and std 1:")
# print(np.random.normal(0, 1, size=5))
# print("\n")
# print("Random array of shape (2,3) from a normal distribution with mean 0 and std 1:")
# print(np.random.normal(0, 1, size=(2,3)))





