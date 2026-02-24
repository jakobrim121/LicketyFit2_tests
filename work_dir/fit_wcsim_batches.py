geometry_path = "/eos/user/j/jrimmer/Geometry"

import sys
sys.path.insert(0, geometry_oath)
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from Geometry.Device import Device
from minuit_fit import *
from LicketyFit.Event import *
from LicketyFit.PMT import *
from LicketyFit.MarkovChain import *
from LicketyFit.Emitter import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import uproot, json, awkward as ak
from read_sim_data import *


cut_time = 15 # Place this cut time right after the initial peak of hit times (i.e. exclude reflections)
energy_true = 300
folder = "/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/mu/1kmu_"+str(energy_true)+"MeV_x0y0zn1000_noScat.npz"
folder = "/eos/user/j/jrimmer/sim_work_dir/WCSim/1kmu_"+str(energy_true)+"MeV_x0y0zn1000_noScatMu_or_ph.npz"
folder = "/eos/user/j/jrimmer/sim_work_dir/WCSim/1kmu_"+str(energy_true)+"MeV_x0y0zn1000.npz"
#folder = "/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/mu/1kmu-_"+str(energy_true)+"MeV_noairgap.npz"
data_raw = read_sim_data(folder)

init_E_vs_dist_travelled = np.load('../tables/init_E_vs_dist_travelled.npy')
dist_travelled = init_E_vs_dist_travelled[:,0]
init_energy = init_E_vs_dist_travelled[:,1]

# Get the mapping between wcsim and wcte PMT positions
wcte_mapping = np.loadtxt('../tables/wcsim_wcte_mapping.txt')

hall = Device.open_file(geometry_path+'/examples/wcte_bldg157.geo')
wcte = hall.wcds[0]
n_mpmt_geom = len(wcte.mpmts)

# wcsim uses positions 1-19, so have to subtract 1 in the mapping...
sim_wcte_mapping = {}
for i in range(len(wcte_mapping)):
    sim_wcte_mapping[int(wcte_mapping[i][0])] = int(wcte_mapping[i][1]*100 + wcte_mapping[i][2] - 1)
    
    
pmts = []

for i in range(len(data_raw['digi_hit_pmt'])):
    for j in range(len(data_raw['digi_hit_pmt'][i])):
        pmts.append(data_raw['digi_hit_pmt'][i][j])
        
        
        
inactive_slots = [27,32,45,74,77,79,85,91,99]
def sim_to_Event(sim_data, n_mpmt_total=None, pe_scale=1.0, shift_times=False):
    """
    Convert one raw JSON event into Event class.

    Parameters
    ----------
    raw : dict
        One decoded JSON entry, containing:
            hit_mpmt_slot_ids
            hit_pmt_position_ids
            hit_pmt_charges
            hit_pmt_times
            run_id
            event_number
    n_mpmt_total : int or None
        If None, inferred from max slot ID + 1.  
        If geometry index space is full (0–105), set this to 106.
    pe_scale : float
        ADC counts per 1 PE (for later use)
    shift_times : bool
        Whether to subtract earliest hit time to make event times relative.

    Returns
    -------
    Event object
    """

    # ---------
    # Determine total number of mPMTs
    # ---------
    slots = []
    pmt_pos_ids = []
    charges = []
    times = []
    for i in range(len(sim_data['digi_hit_pmt'])):
        
        wcte_pmt = sim_wcte_mapping[sim_data['digi_hit_pmt'][i]+1]
        slots.append(int(wcte_pmt/100))
        pmt_pos_ids.append(wcte_pmt%100)
        charges.append(sim_data['digi_hit_charge'][i])
        times.append(sim_data['digi_hit_time'][i])
        
    #slots = raw["hit_mpmt_slot_ids"]
    if n_mpmt_total is None:
        n_mpmt = int(np.max(slots)) + 1
    else:
        n_mpmt = n_mpmt_total

    # Create event
    ev = Event(0, 0, n_mpmt)

    # Activate all PMTs
    ev.set_mpmt_status(list(range(n_mpmt)), True)
    for i_mpmt in range(n_mpmt):
        if i_mpmt in inactive_slots:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), False)
        else:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), True)

    # Fill hits
    for s, p, q, t in zip(slots,
                          pmt_pos_ids,
                          charges,
                          times):
        ev.hit_times[s][p].append(float(t))
        ev.hit_charges[s][p].append(float(q))

    # -------------
    # TIME SHIFTING (this is new)
    # -------------
    if shift_times:
        min_time = float('inf')
        for i_mpmt in range(ev.n_mpmt):
            for i_pmt in range(ev.npmt_per_mpmt):
                if ev.hit_times[i_mpmt][i_pmt]:
                    tmin = min(ev.hit_times[i_mpmt][i_pmt])
                    if tmin < min_time:
                        min_time = tmin

        # Shift all hit times so earliest hit = 0 ns
#         if min_time < float('inf'):
#             for i_mpmt in range(ev.n_mpmt):
#                 for i_pmt in range(ev.npmt_per_mpmt):
#                     ev.hit_times[i_mpmt][i_pmt] = [
#                         t - min_time for t in ev.hit_times[i_mpmt][i_pmt]
#                     ]

        # Store original offset for bookkeeping if needed
        #ev.global_time_offset = min_time

    return ev


def build_observables_from_event(ev, pe_scale=1.0):
    """
    Build obs_pes and obs_ts from a real Event.

    obs_pes: float npe per PMT (from total charge / pe_scale)
    obs_ts:  first hit time per PMT (or None if no hit)

    Returns:
        obs_pes, obs_ts, pmt_indices
        - obs_pes: np.array of len N_pmts_used
        - obs_ts:  np.array of len N_pmts_used (dtype=object so None allowed)
        - pmt_indices: list of (i_mpmt, i_pmt) for each entry
    """
    obs_pes = []
    obs_ts = []
    pmt_indices = []

    for i_mpmt in range(ev.n_mpmt):
        if not ev.mpmt_status[i_mpmt]:
            continue
        for i_pmt in range(ev.npmt_per_mpmt):
            if not ev.pmt_status[i_mpmt][i_pmt]:
                continue

            charges = ev.hit_charges[i_mpmt][i_pmt]
            times   = ev.hit_times[i_mpmt][i_pmt]

            if len(charges) == 0:
                # no hit in this PMT
                obs_pes.append(0.0)
                obs_ts.append(None)
                pmt_indices.append((i_mpmt, i_pmt))
                continue

            # crude PE estimate: total charge / single-PE scale
            total_q = np.sum(charges)
            npe = total_q / pe_scale

            # take earliest hit as "time" of this PMT
            t_first = float(np.min(times))

            obs_pes.append(float(npe))
            obs_ts.append(t_first)
            pmt_indices.append((i_mpmt, i_pmt))

    # use dtype=object so that None is allowed (if you ever keep Nones)
    return np.array(obs_pes, dtype=float), np.array(obs_ts, dtype=object), pmt_indices

# initial guess (it will be varied by MCMC)
starting_time_guess = 0
start_coord_guess   = (0.0, 0.0, 0)   # mm
direction_guess     = (0.0, 0.0, 1.0)   # along +z
beta_guess          = 0.96
length_guess        = 500.0             # mm
intensity_guess     = 18.0               # PE at 1 m, normal incidence

emitter_model = Emitter(starting_time_guess,
                        start_coord_guess,
                        direction_guess,
                        beta_guess,
                        length_guess,
                        intensity_guess)

single_pe_amp_mean = 1.    # arbitrary initial values; tune to match data
single_pe_amp_std  = 0.3
single_pe_time_std = 1.0
separation_time    = 40.
amp_threshold      = 0.2
noise_rate         = 0.

pmt_model = PMT(single_pe_amp_mean,
                single_pe_amp_std,
                single_pe_time_std,
                separation_time,
                amp_threshold,
                noise_rate)

emitter_copy = emitter_model.copy()


corr_pos = {'wut':np.array([]),'delam':np.array([])}
corr_pos = None
def get_neg_log_likelihood_npe_t(x0, y0, z0, cx, cy, length, t0):
    # build direction unit vector
    cz = np.sqrt(1.0 - cx**2 - cy**2)
    direction = (cx, cy, cz)
    start_coord = (x0, y0, z0)

    # update emitter model
    emitter_copy.start_coord   = start_coord
    emitter_copy.starting_time = t0
    emitter_copy.direction     = direction
    emitter_copy.length        = length
    
    init_KE = np.interp(length, dist_travelled*10, init_energy) # Convert distance to mm
    # expected number of PE and times at each PMT
    ss = emitter_copy.get_emission_points(p_locations,init_KE)
    #print(ss[0])
    #print('')
    exp_pes, exp_ts = emitter_copy.get_expected_pes_ts(wcte, ss, p_locations, direction_zs,corr_pos,obs_pes)

    # compare to observed data using the PMT-likelihood model
    neg_ll = pmt_model.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)
    
    # DEBUG GUARD:
#     if not np.isfinite(neg_ll):
#         print("Non-finite neg_ll:",
#               neg_ll,
#               "params:",
#               x0, y0, z0, cx, cy, length, t0)
#         return 1e30

    return float(neg_ll)


est_dict = {'minimum_found':[],'x':[],'y':[],'z':[],'length':[],'t':[],'est_fcn':[],'true_fcn':[],'cx':[],'cy':[]}


fcn = []

tot_events = 1000
#events = []
obs_pes_list = []
obs_ts_list = []

print('Building Events for Multi-processing...')
for evt_num in range(tot_events):
    #print('Event Number',evt_num)
    

    data = {'digi_hit_pmt':[],'digi_hit_time':[],'digi_hit_charge':[]}
    for i in range(len(data_raw['digi_hit_time'][evt_num])):
        if 0<data_raw['digi_hit_time'][evt_num][i]<cut_time:
            data['digi_hit_time'].append(data_raw['digi_hit_time'][evt_num][i])
            data['digi_hit_pmt'].append(data_raw['digi_hit_pmt'][evt_num][i])
            data['digi_hit_charge'].append(data_raw['digi_hit_charge'][evt_num][i])
            

    ev = sim_to_Event(data, n_mpmt_total=106, pe_scale=1.0, shift_times=False)
    #print(ev)
    
    
    p_locations, direction_zs = emitter_copy.get_pmt_placements(ev, wcte, 'design')

    obs_pes, obs_ts, pmt_indices = build_observables_from_event(ev, pe_scale=1.0)
    obs_pes_list.append(obs_pes)
    obs_ts_list.append(obs_ts)
    
    #events.append([obs_pes,obs_ts])


class EventData:
    def __init__(
        self,
        p_locations,      # (N,3)
        direction_zs,     # (N,3)
        obs_pes,          # (N,)
        obs_ts,           # (N,)
        corr_pos,         # correction dict or None
        wcd,
        pmt_model,
        emitter_template  # emitter instance to COPY per fit
    ):
        self.p_locations = np.asarray(p_locations, dtype=float)
        self.direction_zs = np.asarray(direction_zs, dtype=float)
        self.obs_pes = np.asarray(obs_pes, dtype=float)
        self.obs_ts = np.asarray(obs_ts, dtype=float)
        self.corr_pos = corr_pos

        self.wcd = wcd
        self.pmt_model = pmt_model

        # IMPORTANT: each event owns its own emitter instance
        self.emitter = emitter_template.copy()

    def neg_log_likelihood(self, x0, y0, z0, cx, cy, length, t0):
        # Build direction
        cz = np.sqrt(max(0.0, 1.0 - cx**2 - cy**2))
        direction = (cx, cy, cz)

        # Update emitter state
        self.emitter.start_coord = (x0, y0, z0)
        self.emitter.starting_time = t0
        self.emitter.direction = direction
        self.emitter.length = length
        init_KE = np.interp(length, dist_travelled*10, init_energy) # Convert distance to mm
        # Cherenkov geometry
        ss = self.emitter.get_emission_points(self.p_locations,init_KE)

        # Expected PEs and hit times (Model B already inside here)
        exp_pes, exp_ts = self.emitter.get_expected_pes_ts(
            self.wcd,
            ss,
            self.p_locations,
            self.direction_zs,
            self.corr_pos,
            self.obs_pes
        )

        # Likelihood evaluation
        neg_ll = self.pmt_model.get_neg_log_likelihood_npe_t(
            exp_pes,
            self.obs_pes,
            exp_ts,
            self.obs_ts
        )

        return float(neg_ll)


from iminuit import Minuit

def fit_one_event(event, init_params):
    """
    event: EventData instance
    init_params: dict with initial guesses
    """
    m = Minuit(event.neg_log_likelihood, **init_params)
    
    m.limits["x0"] = (-2000, 2000)
    m.limits["y0"] = (-2000, 2000)
    m.limits["z0"] = (-2000, 2000)
    m.limits["cx"] = (-0.5, 0.5)
    m.limits["cy"] = (-0.5, 0.5)
    m.limits["length"] = (0, 3000)   # now FREE
    m.limits["t0"] = (-10, 10)
    
    m.errors["x0"] = 20.0
    m.errors["y0"] = 20.0
    m.errors["z0"] = 20.0
    m.errors["cx"] = 0.01
    m.errors["cy"] = 0.01
    m.errors["length"] = 50.0
    m.errors["t0"] = 0.5

    # Optional configuration
    m.errordef = Minuit.LIKELIHOOD
    m.strategy = 1
    #m.simplex()
    m.migrad(ncall=5000)

    return {
        "values": m.values.to_dict(),
        "errors": m.errors.to_dict(),
        "fval": m.fval,
        "valid": m.valid
    }



from multiprocessing import Pool, cpu_count

def run_all_events(events, init_params, nproc=None):
    if nproc is None:
        nproc = cpu_count()

    args = [(event, init_params) for event in events]

    with Pool(processes=nproc) as pool:
        results = pool.starmap(fit_one_event, args)

    return results


init_E_vs_dist_travelled = np.load('../examples/init_E_vs_dist_travelled.npy')
dist_travelled = init_E_vs_dist_travelled[:,0]
init_energy = init_E_vs_dist_travelled[:,1]

x_t= 0
y_t = 0
z_t = -1000
cx_t = 0
cy_t = 0
length_t = np.interp(energy_true,init_energy,dist_travelled*10)
#print("LENGTH", len)
t_t = 0.

# Build events

n_events_per_batch = 50
for k in range(int(tot_events/n_events_per_batch)):
    
    print('Starting event number', k*50)

    events = []

    for i in range(n_events_per_batch):
        ev = EventData(
            p_locations   = p_locations,
            direction_zs  = direction_zs,
            obs_pes       = obs_pes_list[int(k*n_events_per_batch+i)],
            obs_ts        = obs_ts_list[int(k*n_events_per_batch+i)],
            corr_pos      = corr_pos,
            wcd           = wcte,
            pmt_model     = pmt_model,
            emitter_template = emitter_copy
        )
        events.append(ev)

    # Initial parameter guesses
    init_params = dict(
        x0=0.0,
        y0=0.0,
        z0=-800,
        cx=0.0,
        cy=0.0,
        length=500.0,
        t0=0.0
    )

    # Run fits in parallel
    results = run_all_events(events, init_params, nproc=8)

    
    for j in range(len(results)):
    
        est_dict['minimum_found'].append(int(results[j]['valid']))

        #est_dict['minimum_found'].append(m.fmin)
        est_dict['x'].append(results[j]['values']['x0'])
        est_dict['y'].append(results[j]['values']['y0'])
        est_dict['z'].append(results[j]['values']['z0'])
        est_dict['length'].append(results[j]['values']['length'])
        est_dict['t'].append(results[j]['values']['t0'])
        est_dict['cx'].append(results[j]['values']['cx'])
        est_dict['cy'].append(results[j]['values']['cy'])

        est_dict['est_fcn'].append(results[j]['fval'])
        est_dict['true_fcn'].append(get_neg_log_likelihood_npe_t(
        x_t, y_t, z_t, cx_t, cy_t, length_t, t_t
        ))
    
with open('/eos/user/j/jrimmer/SWAN_projects/beam/e_gamma/LicketyFit/examples/estimates_'+str(energy_true)+'MeV_batches_x0y0zn1000.dict','wb') as f:
    pickle.dump(est_dict,f)
    