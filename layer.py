import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for the heat equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, txy):
        """
        Computing 1st and 2nd derivatives for the heat equation.
        Args:
            tx: input variables (t, x).
        Returns:
            u: network output.
            du_dt: 1st derivative of t.
            du_dx: 1st derivative of x.
            d2u_dt2: 2nd derivative of t.
            d2u_dx2: 2nd derivative of x.
        """

        with tf.GradientTape() as g:
            g.watch(txy)
            with tf.GradientTape() as gg:
                gg.watch(txy)
                u = self.model(txy)
            du_dtxy = gg.batch_jacobian(u, txy)
            du_dt = du_dtxy[..., 0]
            du_dx = du_dtxy[..., 1]
            du_dy = du_dtxy[..., 2]
        d2u_dtxy2 = g.batch_jacobian(du_dtxy, txy)
        d2u_dt2 = d2u_dtxy2[..., 0, 0]
        d2u_dx2 = d2u_dtxy2[..., 1, 1]
        d2u_dy2 = d2u_dtxy2[..., 2, 2]

        return u, du_dt, du_dx, du_dy, d2u_dt2, d2u_dx2, d2u_dy2
