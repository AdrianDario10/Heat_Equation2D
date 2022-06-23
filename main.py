import tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pinn import PINN
from network import Network
from optimizer import L_BFGS_B
from numerical_sol import numerical
from numpy import linalg as LA

def u0(txy):
    """
    Initial wave form.
    Args:
        tx: variables (t, x, y) as tf.Tensor.

    Returns:
        u(t, x, y) as tf.Tensor.
    """

    t = txy[..., 0, None]
    x = txy[..., 1, None]
    y = txy[..., 2, None]


    return   (x**2*(2-x) *  y**2*(2-y))
  
if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """
   
    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 200

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network).build()

    # Time and space domain
    t_f=2
    x_f=2
    x_ini=0
    y_f=2
    y_ini=0
    
    ''' Example of parametric distribution:
    num = (num_train_samples)**(1/3)
    num = int(np.round(num,0))

    epsilon=1e-4
    x=np.linspace(x_ini+epsilon,x_f-epsilon,num)
    t=np.linspace(0+epsilon,t_f-epsilon,num)
    y=np.linspace(y_ini+epsilon,y_f-epsilon,num)

    T, X, Y = np.meshgrid(t,x,y)

    txy_eqn=np.random.rand(num**3, 3)
    txy_eqn[...,0]=T.reshape((num**3,))
    txy_eqn[...,1]=X.reshape((num**3,))
    txy_eqn[...,2]=Y.reshape((num**3,))
    '''
    # create training input
    ## EQUATION
    txy_eqn = np.random.rand(num_train_samples, 3)
    txy_eqn[..., 0] = t_f*txy_eqn[..., 0]               
    txy_eqn[..., 1] = (x_f-x_ini)*txy_eqn[..., 1] + x_ini           
    txy_eqn[..., 2] = (y_f-y_ini)*txy_eqn[..., 2] + y_ini

    ## INITIAL
    txy_ini = np.random.rand(num_train_samples, 3)
    txy_ini[..., 0] = 0                               # t = 0
    txy_ini[..., 1] = (x_f-x_ini)*txy_ini[..., 1] + x_ini            # x = 0 to pi
    txy_ini[..., 2] = (y_f-y_ini)*txy_ini[..., 2] + y_ini

    ## Walls
    txy_low = np.random.rand(num_train_samples, 3)
    txy_low[..., 0] = t_f*txy_low[..., 0]                              # t = 0 to pi
    txy_low[..., 1] = x_ini            # x = 0
    txy_low[..., 2] = (y_f-y_ini)*txy_low[..., 2] + y_ini

    txy_up = np.random.rand(num_train_samples, 3)
    txy_up[..., 0] =t_f*txy_up[..., 0]               # t =  0 to + pi
    txy_up[..., 1] = x_f  # x = + pi
    txy_up[..., 2] = (y_f-y_ini)*txy_up[..., 2] + y_ini

    txy_r = np.random.rand(num_train_samples, 3)
    txy_r[..., 0] = t_f*txy_r[..., 0]                              # t = 0 to pi
    txy_r[..., 1] = (x_f-x_ini)*txy_r[..., 1] + x_ini            # x = 0
    txy_r[..., 2] = y_ini

    txy_l = np.random.rand(num_train_samples, 3)
    txy_l[..., 0] =t_f*txy_l[..., 0]               # t =  0 to + pi
    txy_l[..., 1] = (x_f-x_ini)*txy_l[..., 1] + x_ini  # x = + pi
    txy_l[..., 2] = y_f

    # create training output
    u_zero = np.zeros((num_train_samples, 1))
    u_ones = np.ones((num_train_samples, 1))
    u_ini = u0(tf.constant(txy_ini)).numpy()


    # train the model using L-BFGS-B algorithm
    x_train = [txy_eqn, txy_ini, txy_low, txy_up, txy_r, txy_l]
    y_train = [u_zero, u_ini, u_zero, u_zero, u_zero, u_zero]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # predict u(t,x) distribution
    t_flat = np.linspace(0, t_f, num_test_samples)
    x_flat = np.linspace(x_ini, x_f, num_test_samples)
    y_flat = np.linspace(y_ini, y_f, num_test_samples)
    x, y = np.meshgrid(x_flat, y_flat)
    txy = np.stack([np.zeros((num_test_samples**2, )), x.flatten(), y.flatten()], axis=-1)
    u = network.predict(txy, batch_size=num_test_samples)
    u = u.reshape(x.shape)
    U = numerical(0)
    E0 = U-u

 

    fig= plt.figure(figsize=(15,10))
    vmin, vmax = 0, 1.41
    plt.pcolormesh(x, y, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':40}
    font2 = {'family':'serif','size':15}
    plt.title("t = 0", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(x,y,t)', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=35)
    plt.show()

    ##

    txy = np.stack([np.ones((num_test_samples**2, ))*0.5, x.flatten(), y.flatten()], axis=-1)
    u = network.predict(txy, batch_size=num_test_samples)
    u = u.reshape(x.shape)
    U = numerical(0.5)
    E2 = U-u

    fig= plt.figure(figsize=(15,10))
    vmin, vmax = 0, 1.41
    plt.pcolormesh(x, y, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':40}
    font2 = {'family':'serif','size':15}
    plt.title("t = 0.5", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(x,y,t)', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=35)
    plt.show()

    ##

    txy = np.stack([np.ones((num_test_samples**2, )), x.flatten(), y.flatten()], axis=-1)
    u = network.predict(txy, batch_size=num_test_samples)
    u = u.reshape(x.shape)
    U = numerical(1)
    E3 = U-u

    fig= plt.figure(figsize=(15,10))
    vmin, vmax = 0, 1.41
    plt.pcolormesh(x, y, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':40}
    font2 = {'family':'serif','size':15}
    plt.title("t = 1", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(x,y,t)', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=35)
    plt.show()

    ###################################
    

    fig= plt.figure(figsize=(15,10))
    vmin, vmax = np.min(np.min(E0)), np.max(np.max(E0))
    plt.pcolormesh(x, y, E0, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':40}
    font2 = {'family':'serif','size':15}
    plt.title("t = 0", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('Error', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=35)
    plt.show()
    
    ##

    
    fig= plt.figure(figsize=(15,10))
    vmin, vmax = np.min(np.min(E2)), np.max(np.max(E2))
    plt.pcolormesh(x, y, E2, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':40}
    font2 = {'family':'serif','size':15}
    plt.title("t = 0.5", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('Error', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=35)
    plt.show()

    ##

    fig= plt.figure(figsize=(15,10))
    vmin, vmax = np.min(np.min(E3)), np.max(np.max(E3))
    plt.pcolormesh(x, y, E0, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    font1 = {'family':'serif','size':40}
    font2 = {'family':'serif','size':15}
    plt.title("t = 1", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('Error', fontdict = font1)
    cbar.mappable.set_clim(vmin, vmax)
    cbar.ax.tick_params(labelsize=35)
    plt.show()
