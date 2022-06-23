# Heat_Equation2D
Physics informed neural network (PINN) for the 2D Heat equation

This module implements the Physics Informed Neural Network (PINN) model for the 1D Heat equation. The Transport equation is given by (d/dt - c^2 (d^2/dx^2 + d^2/dy^2))u = 0, where c is 2. It has an initial condition u(x,y,t=0) = x**2(2-x)y**2(2-y). Dirichlet boundary condition is given at x = 0,+2 and y = 0,+2. The PINN model predicts u(x,y,t) for the input (x,y,t).

The effectiveness of PINNs is validated in the following works.

+  M. Raissi, et al., Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, arXiv: 1711.10561 (2017). (https://arxiv.org/abs/1711.10561)

+  M. Raissi, et al., Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations, arXiv: 1711.10566 (2017). (https://arxiv.org/abs/1711.10566)

It is based on hysics informed neural network (PINN) for the 1D Wave equation on https://github.com/okada39/pinn_wave

