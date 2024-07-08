import numpy as np
import pyuvdata
from newcal import calibration_wrappers, caldata, calibration_optimization


data_path = f"/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
model_path = f"/safepool/rbyrne/hera_data/interpolated_models/zen.2459861.45004.sum.abs_calibrated.red_avg_model.uvfits"
data = pyuvdata.UVData()
data.read(data_path)
model = pyuvdata.UVData()
model.read(model_path)

data.inflate_by_redundancy(use_grid_alg=True)
model.inflate_by_redundancy(use_grid_alg=True)

# Model does not include all baselines
model_baselines = list(set(list(zip(model.ant_1_array, model.ant_2_array))))
data_baselines = list(set(list(zip(data.ant_1_array, data.ant_2_array))))
use_baselines = [
    baseline
    for baseline in model_baselines
    if (baseline in data_baselines) or (baseline[::-1] in data_baselines)
]
use_polarizations = -5
data.select(bls=use_baselines, polarizations=use_polarizations)
model.select(bls=use_baselines, polarizations=use_polarizations)

# Align phasing
data.phase_to_time(np.mean(data.time_array))
model.phase_to_time(np.mean(data.time_array))

data.compress_by_redundancy()
model.compress_by_redundancy()

caldata_obj = caldata.CalData()
caldata_obj.load_data(
    data,
    model,
)

avg_spectra = np.load(
    "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
)
delay_spectrum_variance = avg_spectra["variance"]
bl_length_bin_edges = avg_spectra["bl_bin_edges"]
delay_axis = avg_spectra["delay_array"]

calibration_wrappers.get_dwcal_weights_from_delay_spectra(
    caldata_obj,
    delay_spectrum_variance,
    bl_length_bin_edges,
    delay_axis,
)
unflagged_freq_inds = np.where(
    np.sum(caldata_obj.visibility_weights, axis=(0, 1, 3)) > 0
)[0]

print(f"Original abscal params: {caldata_obj.abscal_params}")
initial_cost = calibration_optimization.cost_dw_abscal_wrapper(
    caldata_obj.abscal_params[:, unflagged_freq_inds, 0].flatten(),
    unflagged_freq_inds,
    caldata_obj,
)
print(f"Initial cost: {initial_cost}")

abscal_params_fit = np.load(
    f"/safepool/rbyrne/hera_abscal_Jun2024/zen.2459861.45004.sum.abs_calibrated.red_avg_abscal_params.npy"
)
abscal_params_cost = calibration_optimization.cost_dw_abscal_wrapper(
    abscal_params_fit[:, unflagged_freq_inds, 0].flatten(),
    unflagged_freq_inds,
    caldata_obj,
)
print(f"Abscal cost: {abscal_params_cost}")

calibration_optimization.run_dw_abscal_optimization(
    caldata_obj,
    1e-6,
    1,
    verbose=True,
)

print(caldata_obj.abscal_params)
