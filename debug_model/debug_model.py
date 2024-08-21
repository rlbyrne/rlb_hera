import numpy as np
import pyuvdata
import os

# Get model LSTS
model_filepath = "/safepool/rbyrne/hera_data/H4C-Abscal-Model"
model_filenames = os.listdir(model_filepath)
model_lsts = []
model_lst_filenames = []
for model_file in model_filenames:
    model_uv = pyuvdata.UVData()
    model_uv.read(f"{model_filepath}/{model_file}")
    lsts_new = list(set(model_uv.lst_array))
    model_lsts = np.concatenate((model_lsts, lsts_new))
    filenames_new = np.repeat([model_file], len(lsts_new))
    model_lst_filenames = np.concatenate((model_lst_filenames, filenames_new))

data_uv = pyuvdata.UVData()
data_uv.read(
    "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
)
use_file = True
for use_lst in list(set(data_uv.lst_array)):
    if use_lst <= np.min(model_lsts):
        use_lst += 2 * np.pi
    elif use_lst >= np.max(model_lsts):
        use_lst -= 2 * np.pi
    lst_distance = np.abs(model_lsts - use_lst)
    if np.sort(lst_distance)[1] > 0.0015 / 2:  # Model for the file LST does not exist
        use_file = False

if use_file:  # Keep only files where all LSTs are covered by the model
    model_uv_list = []
    for time_ind, use_lst in enumerate(list(set(data_uv.lst_array))):
        lst_distance = np.abs(model_lsts - use_lst)
        ind1 = np.where(lst_distance == np.min(lst_distance))[0]
        ind2 = np.where(lst_distance == np.sort(lst_distance)[1])[0]
        lst1 = model_lsts[ind1]
        model_filename1 = model_lst_filenames[ind1][0]

        data = data_uv.select(lsts=use_lst, inplace=False)
        model = pyuvdata.UVData()
        model.read(f"{model_filepath}/{model_filename1}")
        model.select(
            lsts=model_lsts[np.where(lst_distance == np.min(lst_distance))[0]],
            inplace=True,
        )

        # Difference data and model
        use_pol = -5
        model.inflate_by_redundancy(use_grid_alg=True)
        data.data_array[np.where(data.flag_array)] = np.nan
        data.inflate_by_redundancy(use_grid_alg=True)
        model_baselines = list(set(list(zip(model.ant_1_array, model.ant_2_array))))
        data_baselines = list(set(list(zip(data.ant_1_array, data.ant_2_array))))
        use_baselines = [
            baseline
            for baseline in model_baselines
            if (baseline in data_baselines) or (baseline[::-1] in data_baselines)
        ]
        data.select(
            bls=use_baselines,
            polarizations=use_pol,
            lsts=data.lst_array[
                np.where(
                    data.lst_array - np.mean(model.lst_array)
                    == np.min(data.lst_array - np.mean(model.lst_array))
                )
            ],
        )
        model.select(bls=use_baselines, polarizations=use_pol)
        data.compress_by_redundancy()
        model.compress_by_redundancy()
        # Align phasing
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))
        # Ensure ordering matches
        data.reorder_blts()
        model.reorder_blts()
        data.reorder_pols(order="AIPS")
        model.reorder_pols(order="AIPS")
        data.reorder_freqs(channel_order="freq")
        model.reorder_freqs(channel_order="freq")
        data.filename = [""]
        model.filename = [""]

        diff = data.sum_vis(
            model,
            difference=True,
            inplace=False,
            override_params=[
                "nsample_array",
                "earth_omega",
                "flag_array",
                "filename",
                "phase_center_catalog",
                "timesys",
                "uvw_array",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "phase_center_frame_pa",
                "vis_units",
                "lst_array",
                "time_array",
                "rdate",
                "dut1",
                "telescope",
                "phase_center_id_array",
                "gst0",
                "flex_spw_id_array",
                "spw_array",
            ],
        )
        diff.flag_array = data.flag_array
        diff.write_uvfits(f"diff_{time_ind}.uvfits")
