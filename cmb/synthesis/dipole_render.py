"""Physics renderer for microbleeds AND their mimics on gradient-echo magnitude images (2026-10-25).

Why: our first synthesis (Arm 1, 2026-09-28) pasted Gaussian blobs and smooth tubes -- negative (-0.130 F1, the tubes were never
confusable with a microbleed). CenSynCMB (arXiv 2607.05325) renders CMB with the susceptibility dipole kernel and adds
'elongated anisotropic masks' as vessel negatives, without saying how the vessels relate to B0. A vein's appearance depends on its
angle to B0 (no extravascular field when parallel, the strongest when perpendicular), which no CMB paper has rendered for training
(literature check 2026-10-25, scratchpad E_render_physics). This module renders three objects with one physics model, so that
labels differ only by geometry:

  sphere    microbleed (chi > 0) or calcification (chi < 0, same magnitude image for the same |chi|: the sign drops out of |S|)
            outside dB/B0 = (chi/3) (a/r)^3 (3 cos^2 t - 1), inside 0 (Lorentz-corrected; e.g. Liu 2012 MRM, DOI 10.1002/mrm.24305)
  cylinder  vein with its axis at angle theta to B0 (Marques & Bowtell 2007, NMR Biomed, DOI 10.1002/nbm.1224, PMID 18078307):
            inside  dB/B0 = (chi/6) (3 cos^2 theta - 1)
            outside dB/B0 = (chi/2) (R/rho)^2 sin^2 theta cos 2phi
            chi of venous blood = Hct [(1 - Y) 3.39 - 0.38] ppm SI (Fan 2013 MRM, DOI 10.1002/mrm.24918, PMID 24006229; 0.27 ppm cgs x 4 pi)

Signal model (static dephasing, magnitude only): on a fine grid c = w exp(-i gamma B0 dB TE), w = 1 in tissue and w_core inside
the object; the acquisition voxel (in-plane size and SLICE THICKNESS) averages c, and the attenuation is A = |mean c| in [0, 1]
(field-free tissue gives 1). A is sampled at the centres of the acquisition voxels (random slab offset) and interpolated linearly
onto the output grid -- the same path a thick 2D slice takes to our 1 x 0.5 x 0.5 mm model grid. Blooming then follows from the
physics (deficit volume grows with TE, B0 and slice thickness; Liu 2012 Radiology, DOI 10.1148/radiol.11110251: x1.5-2.3 from
TE 20 to 40 ms). Host T2/T2* decay is already in the host image; A is applied multiplicatively.

Coordinates are (z, y, x) in mm; B0 points along +z (axial slices) unless b_hat says otherwise.
NOTE for use in training: the stored candidate patches carry FRST and the stage-1 probability as channels 1-2. A rendered object
only changes the IMAGE; both other channels must be recomputed (FRST on the image, probability by the stage-1 U-Net of that fold)
before a rendered crop can stand next to a real candidate -- otherwise the classifier learns 'channels disagree = synthetic'.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

GAMMA = 2.675e8          # rad / s / T


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def field_sphere(z, y, x, centre, radius, chi_ppm, b_hat) -> Tuple[np.ndarray, np.ndarray]:
    """-> (dB/B0 in ppm, inside mask) on the coordinate arrays (mm)."""
    rz, ry, rx = z - centre[0], y - centre[1], x - centre[2]
    r2 = rz * rz + ry * ry + rx * rx
    inside = r2 <= radius * radius
    r2s = np.maximum(r2, 1e-12)
    cos_t2 = (rz * b_hat[0] + ry * b_hat[1] + rx * b_hat[2]) ** 2 / r2s
    field = (chi_ppm / 3.0) * (radius ** 3 / r2s ** 1.5) * (3 * cos_t2 - 1)
    return np.where(inside, 0.0, field), inside


def field_cylinder(z, y, x, point, axis, radius, chi_ppm, b_hat) -> Tuple[np.ndarray, np.ndarray]:
    """Infinite cylinder through `point` along `axis` (unit). -> (dB/B0 in ppm, inside mask)."""
    a = _unit(axis); b = _unit(b_hat)
    pz, py, px = z - point[0], y - point[1], x - point[2]
    along = pz * a[0] + py * a[1] + px * a[2]
    qz, qy, qx = pz - along * a[0], py - along * a[1], px - along * a[2]       # perpendicular part
    rho2 = qz * qz + qy * qy + qx * qx
    inside = rho2 <= radius * radius
    cos_theta = float(np.dot(a, b)); sin2 = 1.0 - cos_theta ** 2
    inner = (chi_ppm / 6.0) * (3 * cos_theta ** 2 - 1)
    if sin2 < 1e-9:                                   # axis parallel to B0: no field outside
        return np.where(inside, inner, 0.0), inside
    bp = b - cos_theta * a; bp = bp / np.linalg.norm(bp)                     # B0 projected into the perpendicular plane
    rho2s = np.maximum(rho2, 1e-12)
    cos_phi2 = (qz * bp[0] + qy * bp[1] + qx * bp[2]) ** 2 / rho2s
    outer = (chi_ppm / 2.0) * (radius ** 2 / rho2s) * sin2 * (2 * cos_phi2 - 1)
    return np.where(inside, inner, outer), inside


def render(shape: Sequence[int], voxel_mm: Sequence[float], obj: Dict, b0_t: float, te_ms: float,
           acq_voxel_mm: Optional[Sequence[float]] = None, b_hat=(1.0, 0.0, 0.0), fine_mm: Sequence[float] = (0.125, 0.125, 0.125),
           slab_offset_mm: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Attenuation map A (shape, float32, 1 = unchanged) on an output grid of `shape` voxels with spacing `voxel_mm` (z, y, x).

    obj: dict(kind='sphere'|'cylinder', centre=(z, y, x) mm in the output frame (cylinder: a point on the axis), radius (mm),
              chi (ppm SI, signed), w_core (signal weight inside: 0 = void core, 1 = no signal loss inside), axis (cylinder, unit)).
    acq_voxel_mm: the acquisition voxel (slice thickness, dy, dx); default = voxel_mm. slab_offset_mm shifts the acquisition
    grid against the object (random in training: avoids grid-locked appearance)."""
    shape = tuple(int(s) for s in shape); voxel = np.asarray(voxel_mm, float)
    acq = np.asarray(acq_voxel_mm if acq_voxel_mm is not None else voxel_mm, float)
    fine = np.asarray(fine_mm, float)
    extent = np.array(shape) * voxel
    margin = acq                                               # one acquisition voxel around the crop
    n = np.ceil((extent + 2 * margin) / fine).astype(int)
    axes = [(-margin[i] + (np.arange(n[i]) + 0.5) * fine[i]) for i in range(3)]
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    b = _unit(b_hat)
    if obj["kind"] == "sphere":
        field, inside = field_sphere(z, y, x, obj["centre"], obj["radius"], obj["chi"], b)
    elif obj["kind"] == "cylinder":
        field, inside = field_cylinder(z, y, x, obj["centre"], obj["axis"], obj["radius"], obj["chi"], b)
    else:
        raise ValueError(obj["kind"])
    phase = GAMMA * b0_t * field * 1e-6 * (te_ms * 1e-3)
    w = np.where(inside, float(obj.get("w_core", 0.0)), 1.0)
    c = w * np.exp(-1j * phase)
    size = np.maximum(np.round(acq / fine).astype(int), 1)
    re = ndimage.uniform_filter(c.real, size=size, mode="nearest"); im = ndimage.uniform_filter(c.imag, size=size, mode="nearest")
    mag = np.hypot(re, im)                                      # |voxel mean| at every fine position
    # acquisition voxel centres covering the crop, shifted by the slab offset; sample the box-averaged magnitude there
    centres = [np.arange(-margin[i] + float(slab_offset_mm[i]) % acq[i] + acq[i] / 2, extent[i] + margin[i], acq[i]) for i in range(3)]
    idx = [np.clip(np.round((cc + margin[i]) / fine[i] - 0.5).astype(int), 0, n[i] - 1) for i, cc in enumerate(centres)]
    a_acq = mag[np.ix_(*idx)]
    # output voxel centres -> fractional index in the acquisition grid -> linear interpolation (thick slice -> model grid)
    out = [(np.arange(shape[i]) + 0.5) * voxel[i] for i in range(3)]
    frac = [np.interp(out[i], centres[i], np.arange(len(centres[i]))) for i in range(3)]
    gz, gy, gx = np.meshgrid(*frac, indexing="ij")
    A = ndimage.map_coordinates(a_acq, [gz, gy, gx], order=1, mode="nearest")
    return np.clip(A, 0.0, 1.0).astype(np.float32)


def deficit_volume(A: np.ndarray, voxel_mm: Sequence[float]) -> float:
    """sum(1 - A) x voxel volume (mm^3): partial-volume-safe apparent size (includes the void core)."""
    return float((1.0 - A).sum() * np.prod(voxel_mm))


def compose_inverted(host_inv: np.ndarray, A: np.ndarray, rng: np.random.RandomState, noise_sd: Optional[float] = None) -> np.ndarray:
    """Apply A to a host crop of our INVERTED images (1 - img / max: microbleeds bright, 0 outside the brain).
    Original magnitude m = 1 - host_inv; new m' = |A m + (1 - A) n| with n Gaussian at the host noise level, so dark voxels keep a
    noise floor instead of becoming perfectly smooth; returns 1 - m' inside the brain, 0 outside."""
    brain = host_inv > 0
    m = 1.0 - host_inv
    if noise_sd is None:
        hp = host_inv - ndimage.gaussian_filter(host_inv, 1.0)
        v = hp[brain]
        noise_sd = 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0
    noise = rng.normal(0.0, noise_sd, m.shape)
    m2 = np.abs(A * m + (1.0 - A) * noise)
    return np.where(brain, np.clip(1.0 - m2, 0.0, 1.0), 0.0).astype(np.float32)


def vein_chi(hct: float, y_sat: float) -> float:
    """ppm SI, Fan 2013 eq. 1 with dchi_do = 3.39 ppm SI per unit Hct and dchi_oxy = -0.38 ppm SI (-0.03 cgs)."""
    return hct * ((1.0 - y_sat) * 3.39 - 0.38)


def sample_object(kind: str, rng: np.random.RandomState, centre) -> Dict:
    """Parameter ranges of the literature check (scratchpad E_render_physics, section 6)."""
    logu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if kind == "bleed":
        while True:
            radius = logu(1.0, 6.0) / 2; chi = logu(0.3, 3.0)
            moment = chi * 4 / 3 * np.pi * radius ** 3
            if 0.5 <= moment <= 25.0:                          # total susceptibility < 25 ppm mm^3 for 92 % of CMB (Liu 2012)
                break
        return dict(kind="sphere", centre=centre, radius=radius, chi=chi, w_core=rng.uniform(0, 0.3))
    if kind == "calcification":
        return dict(kind="sphere", centre=centre, radius=logu(1.0, 5.0) / 2, chi=-logu(0.3, 3.5), w_core=rng.uniform(0, 0.2))
    if kind == "vein":
        axis = _unit(rng.normal(size=3))                       # uniform on the sphere = sin-weighted angle to B0
        return dict(kind="cylinder", centre=centre, axis=axis, radius=logu(0.2, 1.5) / 2,
                    chi=vein_chi(rng.uniform(0.36, 0.46), rng.uniform(0.5, 0.7)), w_core=rng.uniform(0.3, 1.0))
    raise ValueError(kind)


def tilted_b0(rng: np.random.RandomState, max_deg: float = 20.0) -> np.ndarray:
    """B0 along z with a random head tilt up to max_deg."""
    t = np.deg2rad(rng.uniform(0, max_deg)); p = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(t), np.sin(t) * np.sin(p), np.sin(t) * np.cos(p)])
