import torch

from neurodiffeq.conditions import BaseCondition
from neurodiffeq.generators import BaseGenerator
from neurodiffeq.solvers import BaseSolution
from neurodiffeq.neurodiffeq import diff

# Mesh generators for the bundle solver #
class MeshGenerator(BaseGenerator):
    def __init__(self, g1, pg):

        super(MeshGenerator, self).__init__()
        self.g1 = g1
        self.pg = pg

    def get_examples(self):

        u = self.g1.get_examples()
        u = u.reshape(-1, 1, 1)

        bundle_params = self.pg.get_examples()
        if isinstance(bundle_params, torch.Tensor):
            bundle_params = (bundle_params,)
        assert len(bundle_params[0].shape) == 1, "shape error, ask shuheng"
        n_params = len(bundle_params)

        bundle_params = torch.stack(bundle_params, dim=1)
        bundle_params = bundle_params.reshape(1, -1, n_params)

        uu, bb = torch.broadcast_tensors(u, bundle_params)
        uu = uu[:, :, 0].reshape(-1)
        bb = [bb[:, :, i].reshape(-1) for i in range(n_params)]

        return uu, *bb


class SolutionBundle2D(BaseSolution):
    def _compute_u(self, net, condition, *coordinates):
        return condition.enforce(*net, *coordinates)


class BundleIBVP1D(BaseCondition):
    r"""An initial & boundary condition on a 1-D range where :math:`x\in[x_0, x_1]` and time starts at :math:`t_0`.
    The conditions should have the following parts:

    - :math:`u(x,t_0)=u_0(x)`,
    - :math:`u(x_0,t)=g(t)` or :math:`u'_x(x_0,t)=p(t)`,
    - :math:`u(x_1,t)=h(t)` or :math:`u'_x(x_1,t)=q(t)`,

    where :math:`\displaystyle u'_x=\frac{\partial u}{\partial x}`.

    :param x_min: The lower bound of x, the :math:`x_0`.
    :type x_min: float
    :param x_max: The upper bound of x, the :math:`x_1`.
    :type x_max: float
    :param t_min: The initial time, the :math:`t_0`.
    :type t_min: float
    :param t_min_val: The initial condition, the :math:`u_0(x)`.
    :type t_min_val: callable
    :param x_min_val: The Dirichlet boundary condition when :math:`x = x_0`, the :math:`u(x_0, t)`, defaults to None.
    :type x_min_val: callable, optional
    :param x_min_prime: The Neumann boundary condition when :math:`x = x_0`, the :math:`u'_x(x_0, t)`, defaults to None.
    :type x_min_prime: callable, optional
    :param x_max_val: The Dirichlet boundary condition when :math:`x = x_1`, the :math:`u(x_1, t)`, defaults to None.
    :type x_max_val: callable, optional
    :param x_max_prime: The Neumann boundary condition when :math:`x = x_1`, the :math:`u'_x(x_1, t)`, defaults to None.
    :type x_max_prime: callable, optional
    :raises NotImplementedError: When unimplemented boundary conditions are configured.

    .. note::
        This condition cannot be passed to ``neurodiffeq.conditions.EnsembleCondition`` unless both boundaries uses
        Dirichlet conditions (by specifying only ``x_min_val`` and ``x_max_val``) and ``force`` is set to True in
        EnsembleCondition's constructor.
    """

    def __init__(
        self,
        x_min,
        x_max,
        t_min,
        t_min_val,
        x_min_val=None,
        x_min_prime=None,
        x_max_val=None,
        x_max_prime=None,
    ):
        super().__init__()
        n_conditions = sum(
            c is not None for c in [x_min_val, x_min_prime, x_max_val, x_max_prime]
        )
        if (
            n_conditions != 2
            or (x_min_val and x_min_prime)
            or (x_max_val and x_max_prime)
        ):
            raise NotImplementedError(
                "Sorry, this boundary condition is not implemented."
            )
        self.x_min, self.x_min_val, self.x_min_prime = x_min, x_min_val, x_min_prime
        self.x_max, self.x_max_val, self.x_max_prime = x_max, x_max_val, x_max_prime
        self.t_min, self.t_min_val = t_min, t_min_val

    def enforce(self, net, *coordinates):
        r"""Enforces this condition on a network with inputs `x` and `t`

        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param x: The :math:`x`-coordinates of the samples; i.e., the spatial coordinates.
        :type x: `torch.Tensor`
        :param t: The :math:`t`-coordinates of the samples; i.e., the temporal coordinates.
        :type t: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`

        .. note::
            This method overrides the default method of ``neurodiffeq.conditions.BaseCondition`` .
            In general, you should avoid overriding ``enforce`` when implementing custom boundary conditions.
        """
        x = coordinates[0]
        t = coordinates[1]

        def ANN(coordinates):
            # out = net(torch.cat(coordinates, dim=1))
            out = net(torch.stack(coordinates, dim=0).squeeze().t())

            if self.ith_unit is not None:
                out = out[:, self.ith_unit].view(-1, 1)
            return out

        uxt = ANN(coordinates)
        if self.x_min_val and self.x_max_val:
            return self.parameterize(uxt, x, t)
        elif self.x_min_val and self.x_max_prime:
            x1 = self.x_max * torch.ones_like(x, requires_grad=True)
            ux1t = ANN(x1, t)
            return self.parameterize(uxt, x, t, ux1t, x1)
        elif self.x_min_prime and self.x_max_val:
            x0 = self.x_min * torch.ones_like(x, requires_grad=True)
            ux0t = ANN(x0, t)
            return self.parameterize(uxt, x, t, ux0t, x0)
        elif self.x_min_prime and self.x_max_prime:
            x0 = self.x_min * torch.ones_like(x, requires_grad=True)
            x1 = self.x_max * torch.ones_like(x, requires_grad=True)
            ux0t = ANN(x0, t)
            ux1t = ANN(x1, t)
            return self.parameterize(uxt, x, t, ux0t, x0, ux1t, x1)
        else:
            raise NotImplementedError(
                "Sorry, this boundary condition is not implemented."
            )

    def parameterize(self, u, x, t, *additional_tensors):
        r"""Re-parameterizes outputs such that the initial and boundary conditions are satisfied.

        The Initial condition is always :math:`u(x,t_0)=u_0(x)`. There are four boundary conditions that are
        currently implemented:

        - For Dirichlet-Dirichlet boundary condition :math:`u(x_0,t)=g(t)` and :math:`u(x_1,t)=h(t)`:

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\tilde{x}\big(1-\tilde{x}\big)\Big(1-e^{-\tilde{t}}\Big)\mathrm{ANN}(x,t)`,
          where :math:`\displaystyle A(x,t)=u_0(x)+
          \tilde{x}\big(h(t)-h(t_0)\big)+\big(1-\tilde{x}\big)\big(g(t)-g(t_0)\big)`.

        - For Dirichlet-Neumann boundary condition :math:`u(x_0,t)=g(t)` and :math:`u'_x(x_1, t)=q(t)`:

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\tilde{x}\Big(1-e^{-\tilde{t}}\Big)
          \Big(\mathrm{ANN}(x,t)-\big(x_1-x_0\big)\mathrm{ANN}'_x(x_1,t)-\mathrm{ANN}(x_1,t)\Big)`,
          where :math:`\displaystyle A(x,t)=u_0(x)+\big(x-x_0\big)\big(q(t)-q(t_0)\big)+\big(g(t)-g(t_0)\big)`.

        - For Neumann-Dirichlet boundary condition :math:`u'_x(x_0,t)=p(t)` and :math:`u(x_1, t)=h(t)`:

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\big(1-\tilde{x}\big)\Big(1-e^{-\tilde{t}}\Big)
          \Big(\mathrm{ANN}(x,t)-\big(x_1-x_0\big)\mathrm{ANN}'_x(x_0,t)-\mathrm{ANN}(x_0,t)\Big)`,
          where :math:`\displaystyle A(x,t)=u_0(x)+\big(x_1-x\big)\big(p(t)-p(t_0)\big)+\big(h(t)-h(t_0)\big)`.

        - For Neumann-Neumann boundary condition :math:`u'_x(x_0,t)=p(t)` and :math:`u'_x(x_1, t)=q(t)`

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\left(1-e^{-\tilde{t}}\right)
          \Big(
          \mathrm{ANN}(x,t)-\big(x-x_0\big)\mathrm{ANN}'_x(x_0,t)
          +\frac{1}{2}\tilde{x}^2\big(x_1-x_0\big)
          \big(\mathrm{ANN}'_x(x_0,t)-\mathrm{ANN}'_x(x_1,t)\big)
          \Big)`,
          where :math:`\displaystyle A(x,t)=u_0(x)
          -\frac{1}{2}\big(1-\tilde{x}\big)^2\big(x_1-x_0\big)\big(p(t)-p(t_0)\big)
          +\frac{1}{2}\tilde{x}^2\big(x_1-x_0\big)\big(q(t)-q(t_0)\big)`.

        Notations:

        - :math:`\displaystyle\tilde{t}=\frac{t-t_0}{t_1-t_0}`,
        - :math:`\displaystyle\tilde{x}=\frac{x-x_0}{x_1-x_0}`,
        - :math:`\displaystyle\mathrm{ANN}` is the neural network,
        - and :math:`\displaystyle\mathrm{ANN}'_x=\frac{\partial ANN}{\partial x}`.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param x: The :math:`x`-coordinates of the samples; i.e., the spatial coordinates.
        :type x: `torch.Tensor`
        :param t: The :math:`t`-coordinates of the samples; i.e., the temporal coordinates.
        :type t: `torch.Tensor`
        :param additional_tensors: additional tensors that will be passed by ``enforce``
        :type additional_tensors: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """

        t0 = self.t_min * torch.ones_like(t, requires_grad=True)
        x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
        t_tilde = t - self.t_min

        if self.x_min_val and self.x_max_val:
            return self._parameterize_dd(u, x, t, x_tilde, t_tilde, t0)
        elif self.x_min_val and self.x_max_prime:
            return self._parameterize_dn(
                u, x, t, x_tilde, t_tilde, t0, *additional_tensors
            )
        elif self.x_min_prime and self.x_max_val:
            return self._parameterize_nd(
                u, x, t, x_tilde, t_tilde, t0, *additional_tensors
            )
        elif self.x_min_prime and self.x_max_prime:
            return self._parameterize_nn(
                u, x, t, x_tilde, t_tilde, t0, *additional_tensors
            )
        else:
            raise NotImplementedError(
                "Sorry, this boundary condition is not implemented."
            )

    # When we have Dirichlet boundary conditions on both ends of the domain:
    def _parameterize_dd(self, uxt, x, t, x_tilde, t_tilde, t0):
        Axt = (
            self.t_min_val(x)
            + x_tilde * (self.x_max_val(t) - self.x_max_val(t0))
            + (1 - x_tilde) * (self.x_min_val(t) - self.x_min_val(t0))
        )
        return Axt + x_tilde * (1 - x_tilde) * (1 - torch.exp(-t_tilde)) * uxt

    # When we have Dirichlet boundary condition on the left end of the domain
    # and Neumann boundary condition on the right end of the domain:
    def _parameterize_dn(self, uxt, x, t, x_tilde, t_tilde, t0, ux1t, x1):
        Axt = (
            (self.x_min_val(t) - self.x_min_val(t0))
            + self.t_min_val(x)
            + x_tilde
            * (self.x_max - self.x_min)
            * (self.x_max_prime(t) - self.x_max_prime(t0))
        )
        return Axt + x_tilde * (1 - torch.exp(-t_tilde)) * (
            uxt - (self.x_max - self.x_min) * diff(ux1t, x1) - ux1t
        )

    # When we have Neumann boundary condition on the left end of the domain
    # and Dirichlet boundary condition on the right end of the domain:
    def _parameterize_nd(self, uxt, x, t, x_tilde, t_tilde, t0, ux0t, x0):
        Axt = (
            (self.x_max_val(t) - self.x_max_val(t0))
            + self.t_min_val(x)
            + (x_tilde - 1)
            * (self.x_max - self.x_min)
            * (self.x_min_prime(t) - self.x_min_prime(t0))
        )
        return Axt + (1 - x_tilde) * (1 - torch.exp(-t_tilde)) * (
            uxt + (self.x_max - self.x_min) * diff(ux0t, x0) - ux0t
        )

    # When we have Neumann boundary conditions on both ends of the domain:
    def _parameterize_nn(self, uxt, x, t, x_tilde, t_tilde, t0, ux0t, x0, ux1t, x1):
        Axt = (
            self.t_min_val(x)
            - 0.5
            * (1 - x_tilde) ** 2
            * (self.x_max - self.x_min)
            * (self.x_min_prime(t) - self.x_min_prime(t0))
            + 0.5
            * x_tilde**2
            * (self.x_max - self.x_min)
            * (self.x_max_prime(t) - self.x_max_prime(t0))
        )
        return Axt + (1 - torch.exp(-t_tilde)) * (
            uxt
            - x_tilde * (self.x_max - self.x_min) * diff(ux0t, x0)
            + 0.5
            * x_tilde**2
            * (self.x_max - self.x_min)
            * (diff(ux0t, x0) - diff(ux1t, x1))
        )