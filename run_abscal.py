import numpy as np
import pyuvdata
from newcal import calibration_wrappers

def test_abscal_Apr23():

    data_path = "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.52096.sum.abs_calibrated.red_avg.uvh5"
    model_path = "/safepool/rbyrne/hera_data/interpolated_models/zen.2459861.52096.sum.abs_calibrated.red_avg_model.uvfits"
    data = pyuvdata.UVData()
    data.read(data_path)
    model = pyuvdata.UVData()
    model.read(model_path)

    # Abscal requires identical times between the data and the model
    model.time_array = data.time_array

    # Model does not include all baselines
    baselines_model = list(set(list(zip(model.ant_1_array, model.ant_2_array))))
    baselines_data = list(set(list(zip(data.ant_1_array, data.ant_2_array))))
    keep_baselines = [bl for bl in baselines_model if bl in baselines_data or (bl[1], bl[0]) in baselines_data]
    print(len(keep_baselines))
    data.select(bls=keep_baselines)

    abscal_params = calibration_wrappers.absolute_calibration(
        data,
        model,
        log_file_path="/safepool/rbyrne/hera_abscal/calibration_logs/abscal_test_Apr23.txt",
    )
    print(abscal_params)


if __name__ == "__main__":
    test_abscal_Apr23()