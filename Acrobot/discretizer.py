"""Discretización de la observación continua de Acrobot.

La observación es [cos θ1, sin θ1, cos θ2, sin θ2, ω1, ω2]. Hay dos representaciones:
- "angles": θ1 = atan2(sin θ1, cos θ1) y θ2 igual, más ω1 y ω2 (4 dimensiones).
- "raw": las 6 componentes tal cual (6 dimensiones).

Cada dimensión se corta en bins de igual ancho dentro de un rango [low, high].
Los valores fuera del rango caen en el primer o el último bin.
El estado discreto es el índice de la celda (un entero entre 0 y n_states - 1).
"""

import numpy as np

# Límites del entorno (gymnasium, acrobot.py: MAX_VEL_1 = 4π, MAX_VEL_2 = 9π).
OMEGA1_MAX = 4 * np.pi
OMEGA2_MAX = 9 * np.pi

DIM_NAMES = {
    "angles": ["theta1", "theta2", "omega1", "omega2"],
    "raw": ["cos1", "sin1", "cos2", "sin2", "omega1", "omega2"],
}


class Discretizer:
    def __init__(self, representation, bins, ranges):
        """representation: "angles" o "raw".
        bins: cantidad de bins por dimensión (lista, en el orden de DIM_NAMES).
        ranges: (low, high) por dimensión, en el mismo orden.
        """
        if representation not in DIM_NAMES:
            raise ValueError(f"representación desconocida: {representation}")
        n_dims = len(DIM_NAMES[representation])
        if len(bins) != n_dims or len(ranges) != n_dims:
            raise ValueError(f"'{representation}' necesita {n_dims} bins y {n_dims} rangos")
        self.representation = representation
        self.bins = [int(b) for b in bins]
        self.ranges = [(float(lo), float(hi)) for lo, hi in ranges]
        # Bordes interiores: con k bins hay k - 1 cortes. np.digitize devuelve 0..k-1.
        self.edges = [np.linspace(lo, hi, b + 1)[1:-1] for b, (lo, hi) in zip(self.bins, self.ranges)]
        self.n_states = int(np.prod(self.bins))

    def features(self, obs):
        """Pasa la observación cruda a las variables que se discretizan."""
        if self.representation == "angles":
            theta1 = np.arctan2(obs[1], obs[0])
            theta2 = np.arctan2(obs[3], obs[2])
            return np.array([theta1, theta2, obs[4], obs[5]])
        return np.asarray(obs)

    def cell(self, obs):
        """Índice del bin en cada dimensión."""
        x = self.features(obs)
        return tuple(int(np.digitize(x[i], self.edges[i])) for i in range(len(self.bins)))

    def __call__(self, obs):
        """Estado discreto: un único entero que identifica la celda."""
        return int(np.ravel_multi_index(self.cell(obs), self.bins))

    def to_dict(self):
        return {"representation": self.representation, "bins": self.bins, "ranges": self.ranges}

    @classmethod
    def from_dict(cls, d):
        return cls(d["representation"], d["bins"], d["ranges"])


def make_discretizer(representation="angles", bins=10, omega1_max=OMEGA1_MAX, omega2_max=OMEGA2_MAX):
    """Atajo: la misma cantidad de bins en todas las dimensiones (o una lista) y
    rangos simétricos para las velocidades."""
    n_dims = len(DIM_NAMES[representation])
    bins = [bins] * n_dims if np.isscalar(bins) else list(bins)
    if representation == "angles":
        ranges = [(-np.pi, np.pi), (-np.pi, np.pi)]
    else:
        ranges = [(-1.0, 1.0)] * 4
    ranges += [(-omega1_max, omega1_max), (-omega2_max, omega2_max)]
    return Discretizer(representation, bins, ranges)
