import numpy as np

from skfem import MeshLine, Basis, ElementLineP1, BilinearForm, asm, condense, solve
from skfem.helpers import grad

def solve_burgers(r, dt, T, idx, x_left, x_right, dx):
    nsteps = int(T / dt)

    def burgers_profile(x, idx=0):
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

def solve_allen_cahn(r, dt, T, idx, x_left, x_right, dx, reaction_coef = 5):
    nsteps = int(T / dt)

    def ac_profile(x, idx=0):
        return x**2 * np.cos(np.pi * x + idx * np.pi / 6.0)

    def ac_ic(idx=0):
        def f(x):
            return ac_profile(x, idx)
        return f

    def ac_bc_left(x_left, idx=0):
        return ac_profile(np.array([x_left]), idx)[0]

    def ac_bc_right(x_right, idx=0):
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



def solve_fisher_kpp(r, dt, T, idx, x_left, x_right, dx, D=1):
    nsteps = round(T / dt)
    idx += 1

    def fisher_ic_np(x, k):
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