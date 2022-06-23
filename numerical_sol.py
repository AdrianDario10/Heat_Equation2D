import numpy as np
import matplotlib.pyplot as plt

def numerical(T):
  
  axis_size = 200     
  side_length = 2   
  dx = side_length/axis_size   #Space step
  axis_points = np.linspace(0,2,axis_size)   #Spatial grid points for the plate.


  k = 0.25                                     #Thermal diffusivity
  dt = ((1/axis_size)**2)/(2*k)                 #Time step size to ensure a stable discretization scheme.
  n = int(T/dt)                                 #Total number of time steps.

#Create a meshgrid 
  X, Y = np.meshgrid(axis_points, axis_points)

  U = np.zeros((axis_size,axis_size))
#Calculate the initial plate temperature using the temperature initialization function.  This is the initial
#condition of the plate.
  for j in range(axis_size):
    U[j,...] = U[j,...] + axis_points**2*(2 - axis_points) * axis_points[j]**2*(2 - axis_points[j])

#Set up some boundary conditions at the edges 
  U[:,0] = 0
  U[:,-1] = 0
  U[0,:] = 0
  U[-1,:] = 0

# Assign initial boundary conditions to their own variables.
  B1 = U[:,0]
  B2 = U[:,-1]
  B3 = U[0,:]
  B4 = U[-1,:]

# Laplacian numerical approximation using 5-point stencil finite difference methods.
  def laplacian(Z,d_ex):
    Ztop = Z[0:-2,1:-1]
    Zleft = Z[1:-1,0:-2]
    Zbottom = Z[2:,1:-1]
    Zright = Z[1:-1,2:]
    Zcenter = Z[1:-1,1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / d_ex**2

# Now solve the PDE for a result of all spatial positions
#after time T has elapsed.  Iterate over the specified time.

  for i in range(n):

    #Perform the 3rd order differentiation on the function.
    deltaU = laplacian(U,dx)
    
    Uc = U[1:-1,1:-1]

    U[1:-1,1:-1] = Uc + dt * (k*deltaU)

    #Direchlet boundary conditions.  The edges of the plate
    #have steady state, constant temperatures over all time.
    U[:,0] = B1
    U[:,-1] = B2
    U[0,:] = B3
    U[-1,:] = B4


  return U
