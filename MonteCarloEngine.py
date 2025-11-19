"""
sputter_engine.py

Grant Chapman - November 2025

Minimal Monte Carlo sputter deposition engine.

Assumptions:
- Cylindrical chamber.
- Planar circular target at z = 0 facing +z.
- Planar circular substrate at z = H facing -z.
- Uniform neutral gas (constant mean free path).
- Particles emitted from target with cosine angular distribution.
- Collisions = isotropic scattering, optional energy loss per collision.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np



@dataclass
class GeometryParams:
    chamber_radius: float        #[m] radius of cylindrical chamber
    target_radius: float         #[m] radius of target at z = 0
    substrate_radius: float      #[m] radius of substrate at z = H
    target_z: float = 0.0        #[m] z-position of target plane
    substrate_z: float = 0.3     #[m] z-position of substrate plane (distance H)


@dataclass
class GasParams:
    mean_free_path: float        #[m] constant mean free path (can later be function of position)
    energy_loss_per_collision: float = 0.0  #fraction of energy lost per collision (0..1)


@dataclass
class EmissionParams:
    initial_energy_eV: float     #[eV] initial particle energy


@dataclass
class MCParams:
    n_particles: int             #number of Monte Carlo test particles
    max_collisions: int          #max number of collisions before particle is considered lost


@dataclass
class DepositionGridParams:
    n_r: int                     #radial bins on substrate
    n_theta: int                 #angular bins (optional refinement)


def _sample_target_positions(n: int, geom: GeometryParams) -> np.ndarray:
    """
    Sample starting positions uniformly on the circular target surface at z = target_z.
    Returns array of shape (n, 3).
    """
    r_max = geom.target_radius
    #Uniform on disc: r = R*sqrt(u), theta = 2πv
    u = np.random.rand(n)
    v = np.random.rand(n)
    r = r_max * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(n, geom.target_z)
    return np.stack((x, y, z), axis=1)


def _sample_emission_directions_cosine(n: int) -> np.ndarray:
    """
    Sample emission directions with cosine distribution over the upper hemisphere (z > 0).
    Returns array of shape (n, 3) with unit vectors.
    """
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    #Cosine distribution - cos(theta) = sqrt(u1), theta in [0, pi/2]
    cos_theta = np.sqrt(u1)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * u2
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta  #pointing towards +z (towards substrate)
    dirs = np.stack((x, y, z), axis=1)
    #Already unit vectors by construction
    return dirs


def _sample_free_path_length(n: int, mean_free_path: float) -> np.ndarray:
    """
    Sample free path length s from exponential distribution with mean 'mean_free_path'.
    """
    u = np.random.rand(n)
    #s ~ Exp(lambda=1/mean_free_path)
    s = -mean_free_path * np.log(1.0 - u)
    return s


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    """
    Normalize 3D vectors along axis=1.
    """
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return v / norms


def _is_inside_cylinder(xy: np.ndarray, radius: float) -> np.ndarray:
    """
    Check if points (x,y) lie inside radius.
    xy: array (N,2)
    Returns boolean mask of shape (N,).
    """
    r2 = xy[:, 0]**2 + xy[:, 1]**2
    return r2 <= radius**2


def _random_isotropic_directions(n: int) -> np.ndarray:
    """
    Sample n isotropic unit vectors in 3D (for post-collision directions).
    """
    u = np.random.rand(n)
    v = np.random.rand(n)
    cos_theta = 2.0 * u - 1.0  #cos(theta) uniform in [-1,1]
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * v
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    return np.stack((x, y, z), axis=1)

class SputterMonteCarloEngine:
    """
    Core Monte Carlo engine for sputter deposition in a simple cylindrical geometry.
    """

    def __init__(self,
                 geom: GeometryParams,
                 gas: GasParams,
                 emission: EmissionParams,
                 mc: MCParams,
                 grid: DepositionGridParams):
        self.geom = geom
        self.gas = gas
        self.emission = emission
        self.mc = mc
        self.grid = grid

        #Precompute some grid settings for deposition tally
        self._setup_deposition_grid()

    def _setup_deposition_grid(self):
        """
        Prepare the substrate grid in (r, theta) for deposition tally.
        """
        self.r_edges = np.linspace(0.0, self.geom.substrate_radius, self.grid.n_r + 1)
        self.theta_edges = np.linspace(0.0, 2.0 * np.pi, self.grid.n_theta + 1)

        #Accumulator for deposited particles: shape (n_r, n_theta)
        self.deposition_counts = np.zeros((self.grid.n_r, self.grid.n_theta), dtype=np.float64)

    def _deposit_particle(self, hit_pos: np.ndarray):
        """
        Map a substrate hit position to (r, theta) bin and increment deposition count.
        hit_pos: shape (3,), position on substrate plane (z = substrate_z).
        """
        x, y, _ = hit_pos
        r = np.sqrt(x**2 + y**2)
        if r > self.geom.substrate_radius:
            return  #outside substrate, ignore (should be rare if checks are correct)

        theta = np.arctan2(y, x)
        if theta < 0.0:
            theta += 2.0 * np.pi

        #Find bin indices
        i_r = np.searchsorted(self.r_edges, r, side="right") - 1
        i_t = np.searchsorted(self.theta_edges, theta, side="right") - 1

        #Clip in case of edge cases
        i_r = min(max(i_r, 0), self.grid.n_r - 1)
        i_t = min(max(i_t, 0), self.grid.n_theta - 1)

        self.deposition_counts[i_r, i_t] += 1.0

    def _simulate_single_particle(self) -> Tuple[bool, np.ndarray]:
        """
        Simulate one particle from emission to termination.

        Returns:
            (deposited: bool, hit_position: np.ndarray(3,))
            If not deposited, hit_position can be None.
        """
        pos = _sample_target_positions(1, self.geom)[0]
        direction = _sample_emission_directions_cosine(1)[0]
        energy = self.emission.initial_energy_eV

        for _ in range(self.mc.max_collisions):
            #Sample free path
            s = _sample_free_path_length(1, self.gas.mean_free_path)[0]

            #Check intersection with substrate plane z = substrate_z
            z0 = pos[2]
            dz = direction[2]
            hit_substrate = False
            hit_position = None

            if dz > 0.0:
                #distance along direction to reach substrate plane
                s_plane = (self.geom.substrate_z - z0) / dz
                if 0.0 < s_plane <= s:
                    #particle crosses substrate plane within this free flight
                    #compute intersection point
                    hit_pos = pos + s_plane * direction
                    #check if inside substrate radius
                    if np.sqrt(hit_pos[0]**2 + hit_pos[1]**2) <= self.geom.substrate_radius:
                        hit_substrate = True
                        hit_position = hit_pos

            if hit_substrate:
                return True, hit_position

            #No substrate hit in this segment -> move full distance s
            new_pos = pos + s * direction

            #Check if outside chamber radius
            if not _is_inside_cylinder(new_pos[:2].reshape(1, 2), self.geom.chamber_radius)[0]:
                # Particle hit side wall / escaped
                return False, None

            #Check if went behind target (z < target_z) -> consider lost
            if new_pos[2] < self.geom.target_z:
                return False, None

            #Update position
            pos = new_pos

            #Collision occurs at new_pos -> scatter direction
            direction = _random_isotropic_directions(1)[0]
            direction = _normalize_vectors(direction.reshape(1, 3))[0]

            #Energy loss per collision
            if self.gas.energy_loss_per_collision > 0.0:
                energy *= (1.0 - self.gas.energy_loss_per_collision)
                if energy <= 0.0:
                    return False, None

        #Exceeded max collisions without depositing or escaping
        return False, None

    def run(self, seed: int = None) -> Dict[str, np.ndarray]:
        """
        Run the Monte Carlo simulation for n_particles.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            dict with keys:
                - 'deposition_counts': 2D array (n_r, n_theta) of raw hit counts.
                - 'deposition_normalized': normalized by total deposited particles.
                - 'r_edges': radial bin edges.
                - 'theta_edges': angular bin edges.
                - 'num_deposited': total deposited particles.
                - 'num_lost': total lost particles.
        """
        if seed is not None:
            np.random.seed(seed)

        self.deposition_counts.fill(0.0)
        num_deposited = 0
        num_lost = 0

        for _ in range(self.mc.n_particles):
            deposited, hit_pos = self._simulate_single_particle()
            if deposited:
                num_deposited += 1
                self._deposit_particle(hit_pos)
            else:
                num_lost += 1

        if num_deposited > 0:
            deposition_normalized = self.deposition_counts / num_deposited
        else:
            deposition_normalized = np.zeros_like(self.deposition_counts)

        return {
            "deposition_counts": self.deposition_counts.copy(),
            "deposition_normalized": deposition_normalized,
            "r_edges": self.r_edges.copy(),
            "theta_edges": self.theta_edges.copy(),
            "num_deposited": np.array(num_deposited),
            "num_lost": np.array(num_lost),
        }
