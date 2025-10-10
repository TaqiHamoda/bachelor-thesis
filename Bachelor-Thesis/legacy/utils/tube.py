import numpy as np

class Tube:
    def __init__(self, Ls: float, Lc: float, ro: float, ri: float, kappa: float, nu: float, E: float) -> None:
        self.Ls = Ls  # Length of straight section
        self.Lc = Lc  # Length of curved section
        self.ro = ro  # Outer radius of tube
        self.ri = ri  # Inner radius of tube
        self.kappa = kappa  # Curvature of tube
        self.u = [self.kappa, 0, 0]  # Curvature vector

        self.nu = nu  # Poissons Ratio
        self.I = np.pi*(self.ro**4 - self.ri**4)/4  # Moment of Inertia
        self.E = E  # Youngs Modulus
        self.G = self.E/(2 * (1 + self.nu))  # Shear modulus
        self.kbt = np.diag(np.array([self.E*self.I, self.E*self.I, 2*self.G*self.I]))  # Stiffness matrix for bending and torsion

        self.L = self.Ls + self.Lc
