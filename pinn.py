import tensorflow as tf
from layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the heat equation.
    Attributes:
        network: keras network model with input (t, x, y) and output u(t, x, y).
        grads: gradient layer.
    """

    def __init__(self, network, c=0.5):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
        """

        self.network = network
        self.c = c
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the wave equation.
        Returns:
            PINN model for the projectile motion with
                input: [ (t, x, y) relative to equation,
                         (t=0, x, y) relative to initial condition,
                         (t, x=bounds, y=bounds) relative to boundary condition ],
                output: [ u(t,x,y) relative to equation,
                          u(t=0, x,y) relative to initial condition,
                          du_dt(t=0, x,y) relative to initial derivative of t,
                          u(t, x=bounds, y=bounds) relative to boundary condition ]
        """

        # equation input: (t, x)
        txy_eqn = tf.keras.layers.Input(shape=(3,))
        # initial condition input: (t=0, x)
        txy_ini = tf.keras.layers.Input(shape=(3,))
        # boundary
        txy_up = tf.keras.layers.Input(shape=(3,))
        txy_low = tf.keras.layers.Input(shape=(3,))
        txy_r = tf.keras.layers.Input(shape=(3,))
        txy_l = tf.keras.layers.Input(shape=(3,))

        # compute gradients  u, du_dt, du_dx, du_dy, d2u_dt2, d2u_dx2, d2u_dy2
        _, du_dt, _, _, d2u_dt2, d2u_dx2, d2u_dy2 = self.grads(txy_eqn)

        # equation output being zero
        u_eqn = du_dt - self.c * self.c * (d2u_dx2 + d2u_dy2)

        # initial condition output
        u_ini, du_dt_ini, _, _, _, _, _ = self.grads(txy_ini)


        # boundary condition output
        u_up, du_dtUp, _, _, _, d2u_dx2Up, d2u_dy2Up = self.grads(txy_up)   

        u_low, du_dtlow, _, _, _, d2u_dx2low, d2u_dy2low = self.grads(txy_low)

        u_r, u_r_dt, _, _, _, _, _ = self.grads(txy_r) 

        u_l, u_l_dt, _, _, _, _, _ = self.grads(txy_l)


        # build the PINN model for the wave equation
        return tf.keras.models.Model(
            inputs=[txy_eqn, txy_ini, txy_low, txy_up, txy_r, txy_l],
            outputs=[u_eqn, u_ini, u_low, u_up, u_r, u_l])
