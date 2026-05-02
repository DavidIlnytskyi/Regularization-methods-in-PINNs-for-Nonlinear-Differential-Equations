"""Finite-element reference solvers for nonlinear 1D PDEs.

These helpers use scikit-fem to compute numerical reference solutions for the
Burgers, Allen-Cahn, and Fisher-KPP equations. Each public solver returns the
spatial grid, initial condition, and final solution at the requested time.
"""

from typing import Any, TypeAlias

import numpy as np

from skfem import MeshLine, Basis, ElementLineP1, BilinearForm, asm, condense, solve
from skfem.helpers import grad

Array: TypeAlias = np.ndarray[Any, Any]
SolverResult: TypeAlias = tuple[Array, Array, Array]


def solve_burgers(
    r: float,
    dt: float,
    T: float,
    idx: int,
    x_left: float,
    x_right: float,
    dx: float,
) -> SolverResult:
    """Solve the viscous Burgers equation and return ``x``, ``u0``, and ``u(T)``."""
    nsteps = int(T / dt)

    def burgers_profile(x: Array, idx: int = 0) -> Array:
        """Return the sinusoidal initial profile for one head index."""
        return -np.sin(np.pi * x + idx * (np.pi / 6.0))

    mesh = MeshLine(np.arange(x_left, x_right + dx, dx))
    basis = Basis(mesh, ElementLineP1())

    @BilinearForm
    def mass(u, v, w):
        return u * v

    @BilinearForm
    def diffusion(u, v, w):
        return r * grad(u)[0] * grad(v)[0]

    @BilinearForm
    def convection(u, v, w):
        return w.u_prev * grad(u)[0] * v

    M = asm(mass, basis)
    K = asm(diffusion, basis)

    x = basis.doflocs[0]
    u = burgers_profile(x, idx).copy()
    u0 = u.copy()

    D = basis.get_dofs(lambda X: np.isclose(X[0], x_left) | np.isclose(X[0], x_right))

    for n in range(nsteps):
        u_prev_h = basis.interpolate(u)
        C = asm(convection, basis, u_prev=u_prev_h)

        A = M / dt + C + K
        b = M @ (u / dt)

        x_bc = np.zeros(basis.N)
        x_bc[D.flatten()] = 0.0

        AII, bI, xI, I = condense(A, b, D=D, x=x_bc)
        u = xI.copy()
        u[I] = solve(AII, bI)

    return x, u0, u


def solve_allen_cahn(
    r: float,
    dt: float,
    T: float,
    idx: int,
    x_left: float,
    x_right: float,
    dx: float,
    reaction_coef: float = 5,
) -> SolverResult:
    """Solve the Allen-Cahn equation and return ``x``, ``u0``, and ``u(T)``."""
    nsteps = int(T / dt)

    def ac_profile(x: Array, idx: int = 0) -> Array:
        """Return the polynomial-cosine Allen-Cahn profile."""
        return x**2 * np.cos(np.pi * x + idx * np.pi / 6.0)

    def ac_ic(idx: int = 0):
        """Return an initial-condition function for one head index."""

        def f(x: Array) -> Array:
            """Evaluate the initial condition on a grid."""
            return ac_profile(x, idx)

        return f

    def ac_bc_left(x_left: float, idx: int = 0) -> float:
        """Return the left boundary value for one head index."""
        return ac_profile(np.array([x_left]), idx)[0]

    def ac_bc_right(x_right: float, idx: int = 0) -> float:
        """Return the right boundary value for one head index."""
        return ac_profile(np.array([x_right]), idx)[0]

    mesh = MeshLine(np.arange(x_left, x_right + dx, dx))
    basis = Basis(mesh, ElementLineP1())

    @BilinearForm
    def mass(u, v, w):
        return u * v

    @BilinearForm
    def diffusion(u, v, w):
        return r * grad(u)[0] * grad(v)[0]

    M = asm(mass, basis)
    K = asm(diffusion, basis)

    x = basis.doflocs[0]
    u = ac_ic(idx)(x).copy()
    u0 = u.copy()

    left_dofs = basis.get_dofs(lambda X: np.isclose(X[0], x_left))
    right_dofs = basis.get_dofs(lambda X: np.isclose(X[0], x_right))
    D = basis.get_dofs(lambda X: np.isclose(X[0], x_left) | np.isclose(X[0], x_right))

    u_bc = np.zeros(basis.N)
    u_bc[left_dofs.flatten()] = ac_bc_left(x_left, idx)
    u_bc[right_dofs.flatten()] = ac_bc_right(x_right, idx)

    u[left_dofs.flatten()] = u_bc[left_dofs.flatten()]
    u[right_dofs.flatten()] = u_bc[right_dofs.flatten()]

    A = M / dt + K

    for _ in range(nsteps):
        reaction = reaction_coef * (u - u**3)
        b = M @ (u / dt + reaction)
        u = solve(*condense(A, b, D=D, x=u_bc))

    return x, u0, u


def solve_fisher_kpp(
    r: float,
    dt: float,
    T: float,
    idx: int,
    x_left: float,
    x_right: float,
    dx: float,
    D: float = 1,
) -> SolverResult:
    """Solve the Fisher-KPP equation and return ``x``, ``u0``, and ``u(T)``."""
    nsteps = round(T / dt)
    idx += 1

    def fisher_ic_np(x: Array, k: float) -> Array:
        """Return the exponential Fisher-KPP initial condition."""
        xi = (x - x_left) / (x_right - x_left)
        return (np.exp(-k * xi) - np.exp(-k)) / (1.0 - np.exp(-k))

    npoints = int(round((x_right - x_left) / dx)) + 1
    mesh = MeshLine(np.linspace(x_left, x_right, npoints))
    basis = Basis(mesh, ElementLineP1())

    @BilinearForm
    def mass(u, v, w):
        return u * v

    @BilinearForm
    def diffusion(u, v, w):
        return D * grad(u)[0] * grad(v)[0]

    M = asm(mass, basis)
    K = asm(diffusion, basis)

    x = basis.doflocs[0]
    u = fisher_ic_np(x, idx).copy()
    u0 = u.copy()

    left_dofs = basis.get_dofs(lambda X: np.isclose(X[0], x_left))
    right_dofs = basis.get_dofs(lambda X: np.isclose(X[0], x_right))
    bc_dofs = basis.get_dofs(
        lambda X: np.isclose(X[0], x_left) | np.isclose(X[0], x_right)
    )

    u_bc = np.zeros(basis.N)
    u_bc[left_dofs.flatten()] = 1.0
    u_bc[right_dofs.flatten()] = 0.0

    A = M / dt + K

    for _ in range(nsteps):
        reaction = r * u * (1.0 - u)
        b = M @ (u / dt + reaction)
        u = solve(*condense(A, b, D=bc_dofs, x=u_bc))

    return x, u0, u
