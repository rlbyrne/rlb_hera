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

data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
data_filenames = os.listdir(data_filepath)
use_data_filenames = []
for data_file in data_filenames:
    data_uv = pyuvdata.UVData()
    data_uv.read(f"{data_filepath}/{data_file}")
    use_file = True
    for use_lst in list(set(data_uv.lst_array)):
        if use_lst <= np.min(model_lsts):
            use_lst += 2 * np.pi
        elif use_lst >= np.max(model_lsts):
            use_lst -= 2 * np.pi
        lst_distance = np.abs(model_lsts - use_lst)
        if (
            np.sort(lst_distance)[1] > 0.0015 / 2
        ):  # Model for the file LST does not exist
            use_file = False

    if use_file:  # Keep only files where all LSTs are covered by the model
        model_uv_list = []
        for use_lst in list(set(data_uv.lst_array)):
            lst_distance = np.abs(model_lsts - use_lst)
            ind1 = np.where(lst_distance == np.min(lst_distance))[0]
            ind2 = np.where(lst_distance == np.sort(lst_distance)[1])[0]
            lst1 = model_lsts[ind1]
            model_filename1 = model_lst_filenames[ind1][0]
            lst2 = model_lsts[ind2]
            model_filename2 = model_lst_filenames[ind2][0]

            # Interpolate models
            lst_spacing = np.abs(lst2 - lst1)
            lst_spacing1 = np.abs(lst1 - use_lst)
            lst_spacing2 = np.abs(lst2 - use_lst)
            model1_uv = pyuvdata.UVData()
            model1_uv.read(f"{model_filepath}/{model_filename1}")
            model1_uv.select(lsts=[lst1])
            model1_uv.filename = [""]
            model1_uv.data_array *= lst_spacing1 / lst_spacing
            model2_uv = pyuvdata.UVData()
            model2_uv.read(f"{model_filepath}/{model_filename2}")
            model2_uv.select(lsts=[lst2])
            model2_uv.filename = [""]

            # Phase to consistent phase center
            phase_center_time = np.mean(data_uv.time_array)
            model1_uv.phase_to_time(phase_center_time)
            model2_uv.phase_to_time(phase_center_time)

            model2_uv.data_array *= lst_spacing2 / lst_spacing
            model_uv = model1_uv.sum_vis(
                model2_uv,
                inplace=False,
                run_check=False,
                check_extra=False,
                run_check_acceptability=False,
                override_params=["lst_array", "time_array", "uvw_array", "filename"],
            )
            # Correct for decoherence
            model1_uv.data_array = np.abs(model1_uv.data_array) + 0 * 1j
            model2_uv.data_array = np.abs(model2_uv.data_array) + 0 * 1j
            model_uv_abs = model1_uv.sum_vis(
                model2_uv,
                inplace=False,
                run_check=False,
                check_extra=False,
                run_check_acceptability=False,
                override_params=["lst_array", "time_array", "uvw_array", "filename"],
            )
            model_uv.data_array *= np.abs(model_uv_abs.data_array) / np.abs(
                model_uv.data_array
            )
            model_uv_list.append(model_uv)

        # Save output
        combined_model_uv = model_uv_list[0]
        if len(model_uv_list) > 1:
            for model_uv_use in model_uv_list[1:]:
                combined_model_uv.fast_concat(model_uv_use, "blt", inplace=True)
        data_file_name = data_file.removesuffix(".uvh5")
        combined_model_uv.write_uvfits(
            f"/safepool/rbyrne/hera_data/interpolated_models/{data_file_name}_model.uvfits"
        )
