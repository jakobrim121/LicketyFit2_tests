import numpy as np
import matplotlib
import sys
sys.path.insert(0, "/eos/user/j/jrimmer/SWAN_projects/beam/event_display")
sys.path.insert(0, "../")
#matplotlib.use('Agg')

#from Geometry.Device import Device

from scipy.optimize import brentq


c_ang_vs_E = np.load('../tables/mu_cAng_vs_E.npy')
c_ang = c_ang_vs_E[:,0]
energy_for_angle = c_ang_vs_E[:,1]

#dedx_vs_dist_travelled = np.load('./dedx_vs_dist_travelled.npy')
#dedx_table = dedx_vs_dist_travelled[:, 1]
#dist_table = dedx_vs_dist_travelled[:, 0]

E_vs_dist = np.load('../tables/E_vs_dist.npy', allow_pickle=True) # need to convert this from cm to mm later
overall_distances = np.load('../tables/overall_distances.npy')*10 # convert to mm

energy = np.array([a[:, 1] for a in E_vs_dist], dtype=object)
distances = np.array([a[:, 0] for a in E_vs_dist], dtype=object)*10 #change to mm


def cherenkov_scale_muon_water(T_MeV, n=1.33, hard_saturate_above_MeV=None):
    """
    Dimensionless scale factor for Cherenkov light yield vs muon kinetic energy in water.
    Returns 0 below threshold and -> 1 at high energy.

    Parameters
    ----------
    T_MeV : float or ndarray
        Muon kinetic energy [MeV]
    n : float
        Refractive index (default 1.33)
    hard_saturate_above_MeV : float or None
        If set (e.g. 200), force scale = 1 for T >= this value.

    Returns
    -------
    scale : float or ndarray
        Dimensionless factor in [0, 1] (up to numerical clipping)
    """
    m_mu = 105.6583755  # MeV/c^2

    T = np.asarray(T_MeV, dtype=float)
    gamma = (T + m_mu) / m_mu
    beta2 = 1.0 - 1.0 / gamma**2
    beta2 = np.clip(beta2, 0.0, None)

    # Frank–Tamm factor
    ft = 1.0 - 1.0 / (beta2 * n**2)
    ft_inf = 1.0 - 1.0 / (n**2)

    scale = np.zeros_like(T, dtype=float)

    # Cherenkov threshold condition: beta*n > 1  <=>  beta2*n^2 > 1
    mask = beta2 * n**2 > 1.0
    scale[mask] = ft[mask] / ft_inf

    # Clip for numerical safety
    scale = np.clip(scale, 0.0, 1.0)

    # Optional: force saturation above some kinetic energy (your ">=200 MeV => 1" preference)
    if hard_saturate_above_MeV is not None:
        scale[T >= hard_saturate_above_MeV] = 1.0

    return scale




# def theta_c_func(angles, E, E_k):
    
#     angles = np.asarray(angles)
#     E = np.asarray(E)

#     if E_k < E.min() or E_k > E.max():
#         raise ValueError("E_k is outside the interpolation range")
        
#     angle = np.interp(E_k, E, angles)

#     return angle

import numpy as np

def theta_c_func(angles, E, E_k):
    """
    Vectorized Cherenkov angle interpolation.

    Parameters
    ----------
    angles : array-like, shape (M,)
        Cherenkov angles (radians) tabulated vs energy
    E : array-like, shape (M,)
        Energies corresponding to `angles` (MeV), must be sorted
    E_k : float or ndarray
        Kinetic energy (MeV) at which to evaluate theta_c

    Returns
    -------
    theta_c : float or ndarray
        Interpolated Cherenkov angle(s), same shape as E_k
    """

    angles = np.asarray(angles, dtype=float)
    E = np.asarray(E, dtype=float)
    E_k = np.asarray(E_k, dtype=float)

    # Vectorized bounds check
    if np.any(E_k < E.min()) or np.any(E_k > E.max()):
        print(E_k)
        raise ValueError("One or more E_k values are outside the interpolation range")

    # np.interp is already vectorized in the x argument
    theta_c = np.interp(E_k, E, angles)

    return theta_c



'''

def find_scale_for_pmts(
    pmt_pos, start_pos, track_dir,
    s_a_mm, s_max_mm,
    theta_c_func,
    mpmt_bool,
    R_pmt_mm=37.5,
    n_scan=600,
):
    """
    Band-limited Cherenkov-cone-collapse scaling + return s_b (first intersection).

    Returns
    -------
    scale_eff : (N,) float
        Average Cherenkov yield scale over the contributing (band) region.
        0 if no contributing region found on the grid.
    L_eff_mm : (N,) float
        Effective contributing track length (mm) over which the band condition holds.
    s_b : (N,) float
        First Cherenkov "hit" distance along track (mm), defined as the first s where
        f(s)=alpha(s)-theta_c(s) crosses from negative to non-negative:
            crossing = (f[:, :-1] < 0) & (f[:, 1:] >= 0)
        NaN if no such crossing is found.
    """

    pmt_pos   = np.asarray(pmt_pos, float)
    start_pos = np.asarray(start_pos, float)
    track_dir = np.asarray(track_dir, float)
    track_dir = track_dir / np.linalg.norm(track_dir)

    N = pmt_pos.shape[0]

    # ---- Choose closest precomputed E(s) table (your existing logic) ----
    main_idx = np.searchsorted(overall_distances, s_max_mm)
    main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)
    left = overall_distances[main_idx - 1]
    right = overall_distances[main_idx]
    main_idx -= (s_max_mm - left) <= (right - s_max_mm)

    # ---- s grid ----
    s_grid = np.linspace(s_a_mm, s_max_mm, n_scan).astype(float)   # (S,)
    ds_mm  = s_grid - s_a_mm

    # ---- E(s) ----
    idx = np.searchsorted(distances[main_idx], ds_mm)
    idx = np.clip(idx, 1, len(distances[main_idx]) - 1)
    E_grid = energy[main_idx][idx].astype(float)
    E_grid = np.maximum(E_grid, 54.45)

    # ---- theta_c(s), yield scale y(s) ----
    theta_c_grid = theta_c_func(c_ang, energy_for_angle, E_grid)   # (S,)
    y_grid = cherenkov_scale_muon_water(E_grid)                    # (S,)

    # ---- emission points ----
    emit_pos = start_pos + np.outer(s_grid, track_dir)             # (S,3)

    # ---- geometry to PMTs ----
    d = pmt_pos[:, None, :] - emit_pos[None, :, :]                 # (N,S,3)
    r = np.linalg.norm(d, axis=2)                                  # (N,S)
    r = np.where(r == 0, 1e-9, r)

    cos_alpha = np.einsum("ijk,k->ij", d, track_dir) / r
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)                                   # (N,S)

    # ---- Cherenkov matching function ----
    f = alpha - theta_c_grid[None, :]                               # (N,S)

    # ---- s_b: first "exit" crossing (same convention you used) ----
    s_b = np.full(N, np.nan, dtype=float)
    crossing = (f[:, :-1] < 0.0) & (f[:, 1:] >= 0.0)               # (N,S-1)
    has_cross = crossing.any(axis=1)
    if np.any(has_cross):
        first_idx = np.argmax(crossing[has_cross], axis=1)         # first True per row
        s_b[has_cross] = s_grid[first_idx]

    # ---- Finite-PMT angular band around f=0 ----
    # Exact disk half-angle is asin(R/r); for small angles asin(x)≈x -> R/r
    dalpha = R_pmt_mm / r # (N,S)
    #dalpha_exact = np.arcsin(R_pmt_mm/r)
    
    
    band = np.abs(f) <= dalpha                                      # (N,S)

    # ---- Integrate yield over band (trapezoid-like, segment-based) ----
    ds = np.diff(s_grid)                                            # (S-1,)
    y_mid = 0.5 * (y_grid[:-1] + y_grid[1:])                        # (S-1,)

    band_seg = band[:, :-1] & band[:, 1:]                           # (N,S-1)

    num = (band_seg * y_mid[None, :]) * ds[None, :]                 # (N,S-1)
    den = (band_seg.astype(float)) * ds[None, :]                    # (N,S-1)

    num_int = np.sum(num, axis=1)                                   # (N,)
    den_int = np.sum(den, axis=1)                                   # (N,) effective length

    scale_eff = np.zeros(N, dtype=float)
    ok = den_int > 0
    scale_eff[ok] = num_int[ok] / den_int[ok]
    
    if mpmt_bool:
  
        #print('dEdx_MeV_per_cm',dEdx_MeV_per_cm)
        print('ds_cm',ds_mm)
        print('E_grid',E_grid)
        #print('E_b',E_b)
        print('s_b',s_b)
        #print('alpha',alpha)
        #print('theta',theta_c_grid[None, :])
        print('f', f)
        print('dalpha',dalpha)
        print('scale_eff',scale_eff)
        print('band',band)
        print('den_int',den_int)
        #print('dalpha ratio',dalpha/dalpha_exact)
        

    return scale_eff, s_b




'''
def find_scale_for_pmts(
    pmt_pos,                 # (N, 3)
    start_pos,               # (3,)
    track_dir,               # (3,) unit vector
    s_a_mm,
    s_max_mm,
    theta_c_func,
    mpmt_bool=False,
    n_scan=150,
):
    """
    Vectorized Cherenkov-cone-collapse scale finder for many PMTs.

    SAME LOGIC as your current implementation:
      - compute f(s)=alpha(s)-theta_c(s) on an s-grid
      - find first crossing where f goes from <0 to >=0
      - compute E_b at s_b (nearest lookup)
      - scale[has_crossing] = cherenkov_scale_muon_water(E_b)
      - return (scale, s_b)

    Optimizations:
      - no (N,S,3) tensor allocation
      - no arccos; alpha computed via atan2(sqrt(perp2), parallel) exactly
    """

    import numpy as np

    # ---------- Setup ----------
    pmt_pos   = np.asarray(pmt_pos, dtype=float)        # (N,3)
    start_pos = np.asarray(start_pos, dtype=float)      # (3,)
    track_dir = np.asarray(track_dir, dtype=float)      # (3,)
    track_dir = track_dir / np.linalg.norm(track_dir)

    N = pmt_pos.shape[0]

    # ---- Choose closest precomputed E(s) table (same as yours) ----
    main_idx = np.searchsorted(overall_distances, s_max_mm)
    main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)
    left = overall_distances[main_idx - 1]
    right = overall_distances[main_idx]
    main_idx -= (s_max_mm - left) <= (right - s_max_mm)

    # ---------- s grid ----------
    s_grid = np.linspace(s_a_mm, s_max_mm, n_scan)      # (S,)
    ds_mm  = (s_grid - s_a_mm)                          # (S,)

    idx = np.searchsorted(distances[main_idx], ds_mm)
    idx = np.clip(idx, 1, len(distances[main_idx]) - 1)

    E_grid = energy[main_idx][idx]
    E_grid = np.maximum(E_grid, 54.45)                  # enforce threshold

    theta_c_grid = theta_c_func(c_ang, energy_for_angle, E_grid)  # (S,)

    # ---------- Emission points ----------
    emit_pos = start_pos + s_grid[:, None] * track_dir[None, :]    # (S,3)

    # ---------- Geometry without d(N,S,3) ----------
    # r^2 = |p - e|^2 = |p|^2 + |e|^2 - 2 p·e
    p_norm2 = np.einsum("ij,ij->i", pmt_pos, pmt_pos)             # (N,)
    e_norm2 = np.einsum("ij,ij->i", emit_pos, emit_pos)           # (S,)
    p_dot_e = pmt_pos @ emit_pos.T                                 # (N,S)

    r2 = p_norm2[:, None] + e_norm2[None, :] - 2.0 * p_dot_e      # (N,S)
    r2 = np.maximum(r2, 1e-12)

    # parallel = (p - e)·t = p·t - e·t
    p_dot_t = pmt_pos @ track_dir                                  # (N,)
    e_dot_t = emit_pos @ track_dir                                 # (S,)
    parallel = p_dot_t[:, None] - e_dot_t[None, :]                 # (N,S)

    # perp^2 = r^2 - parallel^2
    perp2 = r2 - parallel * parallel
    perp2 = np.maximum(perp2, 0.0)

    # alpha in [0,pi]: exact equivalent to arccos(parallel / r)
    alpha = np.arctan2(np.sqrt(perp2), parallel)                   # (N,S)

    # ---------- f(s) = alpha - theta_c ----------
    f = alpha - theta_c_grid[None, :]                               # (N,S)

    # ---------- Find first crossing (same logic) ----------
    scale = np.zeros(N, dtype=float)
    crossing = (f[:, :-1] < 0.0) & (f[:, 1:] >= 0.0)               # (N,S-1)
    has_crossing = crossing.any(axis=1)                             # (N,)

    # NOTE: original code does this even for rows with no crossing
    first_idx = np.argmax(crossing, axis=1)                         # (N,)
    s_b = s_grid[first_idx]                                         # (N,)

    # ---------- Energy at s_b (same nearest-neighbor logic) ----------
    ds_mm_b = (s_b - s_a_mm)

    idx_b = np.searchsorted(distances[main_idx], ds_mm_b)
    idx_b = np.clip(idx_b, 1, len(distances[main_idx]) - 1)

    left_b = distances[main_idx][idx_b - 1]
    right_b = distances[main_idx][idx_b]
    choose_right_b = (ds_mm_b - left_b) > (right_b - ds_mm_b)

    nearest_idx_b = idx_b.copy()
    nearest_idx_b[~choose_right_b] = idx_b[~choose_right_b] - 1

    E_b = energy[main_idx][nearest_idx_b]

    # ---------- Scale only for PMTs that actually cross ----------
    scale[has_crossing] = cherenkov_scale_muon_water(E_b[has_crossing])

    return scale, s_b



def find_scale_for_pmts_old(
    pmt_pos,                 # (N, 3)
    start_pos,               # (3,)
    track_dir,               # (3,) unit vector
    s_a_mm,
    s_max_mm,
    #E_at_sa_MeV,
    theta_c_func,
    #mpmt_bool,
    n_scan= 150
):
    """
    Vectorized Cherenkov-cone-collapse scale finder for many PMTs.

    Returns
    -------
    scale : ndarray, shape (N,)
        Cherenkov light scale factor for each PMT (0 if never illuminated)
    """

    # ---------- Setup ----------
    pmt_pos   = np.asarray(pmt_pos, dtype=float)        # (N,3)
    start_pos = np.asarray(start_pos, dtype=float)
    track_dir = np.asarray(track_dir, dtype=float)

    track_dir = track_dir / np.linalg.norm(track_dir)

    N = pmt_pos.shape[0]
    
    
    
   
    main_idx = np.searchsorted(overall_distances, s_max_mm)

    # keep indices in range
    main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)

    # choose nearest of left/right neighbors
    left = overall_distances[main_idx - 1]
    right = overall_distances[main_idx]

    main_idx -= (s_max_mm - left) <= (right - s_max_mm)


    # ---------- s grid ----------
    s_grid = np.linspace(s_a_mm, s_max_mm, n_scan)      # (S,)
    ds_mm  = (s_grid - s_a_mm)  

    idx = np.searchsorted(distances[main_idx], ds_mm)
    idx = np.clip(idx, 1, len(distances[main_idx]) - 1)
    
    E_grid = energy[main_idx][idx]
    E_grid = np.maximum(E_grid, 54.45) # enforce threshold
    
#     s_grid = np.linspace(s_a_mm, s_max_mm, n_scan)      # (S,)
#     ds_mm  = (s_grid - s_a_mm) 
    
#     idx = np.searchsorted(dist_table, ds_mm)
#     idx = np.clip(idx, 1, len(dist_table) - 1)
    
#     # Pick the nearer of dist_table[idx-1] and dist_table[idx]
#     left = dist_table[idx - 1]
#     right = dist_table[idx]
#     choose_right = (ds_mm - left) > (right - ds_mm)   # True if right is closer (ties go left)

#     nearest_idx = idx.copy()
#     nearest_idx[~choose_right] = idx[~choose_right] - 1

    # Lookup dE/dx for each ds_cm
#     dEdx_MeV_per_cm = dedx_table[nearest_idx]   # shape (S,)

    # Energy along track
    #E_grid = E_at_sa_MeV - dEdx_MeV_per_cm * ds_cm
    #E_grid = E_at_sa_MeV - dEdx_MeV_per_cm * ds_cm
    
    

    theta_c_grid = theta_c_func(c_ang, energy_for_angle, E_grid)  # (S,)
    

    # ---------- Emission points ----------
    emit_pos = start_pos + np.outer(s_grid, track_dir)  # (S,3)

    # ---------- Geometry: PMT vectors ----------
    # d[i,j,:] = vector from emission point j to PMT i
    d = pmt_pos[:, None, :] - emit_pos[None, :, :]      # (N,S,3)

    r = np.linalg.norm(d, axis=2)                        # (N,S)
    valid = r > 0
    #print('d',d[0])
    #print('r',r[0])
    # Photon angle wrt track
    cos_alpha = np.zeros_like(r)
    cos_alpha[valid] = np.einsum("ijk,k->ij", d, track_dir)[valid] / r[valid]
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    alpha = np.arccos(cos_alpha)                         # (N,S)
    #print('alpha',alpha[0])
    # ---------- f(s) = alpha - theta_c ----------
    f = alpha - theta_c_grid[None, :]# (N,S)
    #print('f',f[0])
    #print('theta', theta_c_grid[None, :])

    # ---------- Find first crossing ----------
    #illuminated = f <= 0   # (N,S)
    #print(illuminated)

    scale = np.zeros(N)
    crossing = (f[:, :-1] < 0) & (f[:, 1:] >= 0)# shape (N, S-1)
    #print('crossing',crossing[0])
    has_crossing = crossing.any(axis=1)            # shape (N,)
    #print('has crossing',has_crossing)

    #has_crossing = illuminated.any(axis=1)
    #print(has_crossing)
    first_idx = np.argmax(crossing, axis=1)        # shape (N,)
    #first_idx = np.argmax(illuminated, axis=1)           # first True per row
    #print('first index',first_idx)
    s_b = s_grid[first_idx]                               # (N,)

    # ---------- Energy at s_b ----------
    ds_mm_b = (s_b - s_a_mm) 
    
    idx_b = np.searchsorted(distances[main_idx], ds_mm_b)
    idx_b = np.clip(idx_b, 1, len(distances[main_idx]) - 1)
    
    # Pick the nearer of dist_table[idx-1] and dist_table[idx]
    left_b = distances[main_idx][idx_b - 1]
    right_b = distances[main_idx][idx_b]
    choose_right_b = (ds_mm_b - left_b) > (right_b - ds_mm_b)   # True if right is closer (ties go left)

    nearest_idx_b = idx_b.copy()
    nearest_idx_b[~choose_right_b] = idx_b[~choose_right_b] - 1
    
    E_b = energy[main_idx][nearest_idx_b]

    # Lookup dE/dx for each ds_cm
    #dEdx_MeV_per_cm_b = dedx_table[nearest_idx_b]   # shape (S,)
    
    #E_b = E_at_sa_MeV - dEdx_MeV_per_cm_b * ds_cm_b

    scale[has_crossing] = cherenkov_scale_muon_water(E_b[has_crossing])
    
    
    
    
    ### FOLLOWING CODE IS FOR DEBUGGING ###
    
    try:
    
#         s_b_p1 = s_grid[first_idx+1]                               # (N,)

#         # ---------- Energy at s_b ----------
#         ds_mm_b_p1 = (s_b_p1 - s_a_mm) 

#         idx_b_p1 = np.searchsorted(distances[main_idx], ds_mm_b_p1)
#         idx_b_p1 = np.clip(idx_b_p1, 1, len(distances[main_idx]) - 1)

#         # Pick the nearer of dist_table[idx-1] and dist_table[idx]
#         left_b_p1 = distances[main_idx][idx_b_p1 - 1]
#         right_b_p1 = distances[main_idx][idx_b_p1]
#         choose_right_b_p1 = (ds_mm_b_p1 - left_b_p1) > (right_b_p1 - ds_mm_b_p1)   # True if right is closer (ties go left)

#         nearest_idx_b_p1 = idx_b_p1.copy()
#         nearest_idx_b_p1[~choose_right_b_p1] = idx_b_p1[~choose_right_b_p1] - 1

#         E_b_p1 = energy[main_idx][nearest_idx_b_p1]

#         scale_p1 = np.zeros(N)

#         scale_p1[has_crossing] = cherenkov_scale_muon_water(E_b_p1[has_crossing])

#         if mpmt_bool:
#             #print('E_at_sa_MeV',E_at_sa_MeV)
#             #print('dEdx_MeV_per_cm',dEdx_MeV_per_cm)
#             print('ds_cm',ds_mm)
#             print('E_grid',E_grid)
#             print('E_b',E_b)
#             print('s_b',s_b)
#             #print('alpha',alpha)
#             #print('theta',theta_c_grid[None, :])
#             print('f', f)
#             print('scale',scale)
#             print('DISTANCE BEFORE STOP:', 1165-s_b)
            
            #print('has_crossing',has_crossing)
#             print('first_idx',first_idx)
#             print('s_b_p1',s_b_p1)
#             print('E_b_p1',E_b_p1)
#             print('scale_p1',scale_p1)
            print('')
            
    except:
        #print('Crap')
        pass

    return scale, s_b


'''

# def find_scale_for_pmt(
#     pmt_pos,
#     start_pos,
#     track_dir,
#     s_a_mm,
#     s_max_mm,
#     E_at_sa_MeV,
#     dEdx_MeV_per_cm,
#     theta_c_func
# ):
#     """
#     Find s_b: the earliest point along the track where the collapsing
#     Cherenkov cone reaches the PMT.

#     Parameters
#     ----------
#     pmt_pos : array-like, shape (3,)
#         PMT position [mm]
#     start_pos : array-like, shape (3,)
#         Track start position [mm]
#     track_dir : array-like, shape (3,)
#         Unit vector along track direction
#     s_a_mm : float
#         s value where cone collapse begins [mm]
#     s_max_mm : float
#         Track end [mm]
#     E_at_sa_MeV : float
#         Muon kinetic energy at s = s_a [MeV]
#     dEdx_MeV_per_cm : float
#         Energy loss rate (positive number) [MeV/cm]
#     theta_c_func : callable
#         theta_c_func(E_MeV) -> Cherenkov angle [rad]

#     Returns
#     -------
#     s_b_mm : float or None
#         s value where PMT first becomes illuminated,
#         or None if no solution exists in [s_a, s_max]
#     """

#     pmt_pos = np.asarray(pmt_pos, dtype=float)
#     start_pos = np.asarray(start_pos, dtype=float)
#     track_dir = np.asarray(track_dir, dtype=float)

#     def energy_at_s(s_mm):
#         ds_cm = (s_mm - s_a_mm) / 10.0
#         return E_at_sa_MeV - dEdx_MeV_per_cm * ds_cm

#     def alpha_minus_theta(s_mm):
#         # Emission point
#         emit_pos = start_pos + s_mm * track_dir

#         # Vector from emission point to PMT
#         d = pmt_pos - emit_pos
#         r = np.linalg.norm(d)

#         if r == 0:
#             return np.inf

#         # Angle between photon direction and track
        
#         cos_alpha = np.dot(track_dir, d) / r
#         cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
#         alpha = np.arccos(cos_alpha)

#         E = energy_at_s(s_mm)
#         if E <= 0:
#             return np.inf
#         if E<54.4:
#             E = 54.45
       

#         theta_c = theta_c_func(c_ang,energy,E)
        

#         return alpha - theta_c

#     # Check if a solution is bracketed
#     f_a = alpha_minus_theta(s_a_mm)
#     f_b = alpha_minus_theta(s_max_mm)
    
#     #print('fa',f_a)
#     #print('fb',f_b)
   
#     #if f_a > 0 and f_b > 0:
#         # Cone never wide enough
#         #return None

#     try:
#         s_b = brentq(alpha_minus_theta, s_a_mm, s_max_mm)
#         #print('sb',s_b)
#         #return s_b
    
#     except:
#         # If no zero crossing is found, check if there are two zero crossings
#         try:
#             #print('ENTERED SECOND ATTEMPT')
#             n_scan = 200
#             ss = np.linspace(s_a_mm, s_max_mm, n_scan)
#             fs = np.array([alpha_minus_theta(s) for s in ss])
            

#             # ignore NaNs/Infs
#             good = np.isfinite(fs)
#             ss, fs = ss[good], fs[good]
#             if len(ss) < 2:
#                 return None
            
#             #print('SS LENGTH > 2')

#             # find first sign change
#             sign = np.sign(fs)
#             idx = np.where(sign[:-1] * sign[1:] < 0)[0]
#             if len(idx) == 0:
#                 return None
            
#             #print('IDX LENGTH NOT 0')

#             i = idx[0]
#             s_b = brentq(alpha_minus_theta, ss[i], ss[i+1])
            
#             E_at_sb =  E_at_sa_MeV - (dEdx_MeV_per_cm/10.0)*(s_b-s_a_mm)
            
#             return cherenkov_scale_muon_water(E_at_sb)
        
#         except:
    
        
#             return None
    

         
#     E_at_sb =  E_at_sa_MeV - (dEdx_MeV_per_cm/10.0)*(s_b-s_a_mm)
#     #print('Energy',E_at_sb)
    
#     return cherenkov_scale_muon_water(E_at_sb)
'''