import numpy as np
import pyuvdata
from newcal import calibration_wrappers
import sys

def test_abscal_Apr23():

    datafile_name = "zen.2459861.52096.sum.abs_calibrated.red_avg"
    data_path = f"/safepool/rbyrne/hera_data/H6C-data/2459861/{datafile_name}.uvh5"
    model_path = f"/safepool/rbyrne/hera_data/interpolated_models/{datafile_name}_model.uvfits"
    data = pyuvdata.UVData()
    data.read(data_path)
    model = pyuvdata.UVData()
    model.read(model_path)

    data.inflate_by_redundancy(use_grid_alg=True)
    model.inflate_by_redundancy(use_grid_alg=True)

    # Model does not include all baselines
    model_baselines = list(set(list(zip(model.ant_1_array, model.ant_2_array))))
    data_baselines = list(set(list(zip(data.ant_1_array, data.ant_2_array))))
    use_baselines = [baseline for baseline in model_baselines if (baseline in data_baselines) or (baseline[::-1] in data_baselines)]
    use_polarizations = [pol for pol in model.polarization_array if pol in data.polarization_array]
    data.select(bls=use_baselines, polarizations=use_polarizations)
    model.select(bls=use_baselines, polarizations=use_polarizations)

    # Align phasing
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))

    #null = data.sum_vis(model, inplace=False, override_params=["nsample_array", "earth_omega", "flag_array", "filename", "phase_center_catalog", "timesys"])  # Verify that metadata match

    abscal_params = calibration_wrappers.absolute_calibration(
        data,
        model,
        log_file_path="/safepool/rbyrne/hera_abscal/calibration_logs/abscal_test_May3.txt",
    )
    print(abscal_params[0,:,0])

    data = pyuvdata.UVData()
    data.read(data_path)
    data.phase_to_time(np.mean(data.time_array))
    calibration_wrappers.apply_abscal(data, abscal_params, data.polarization_array, inplace=True)
    data.write_uvfits(f"/safepool/rbyrne/hera_abscal/{datafile_name}_abscal.uvfits")


if __name__ == "__main__":
    test_abscal_Apr23()