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

# Check if model LSTS cover data range
data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
data_filenames = os.listdir(data_filepath)
use_data_filenames = []
for data_file in data_filenames:
    data_uv = pyuvdata.UVData()
    data_uv.read(data_file)
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
        use_data_filenames.append(data_file)


for data_file in use_data_filenames:
    data_uv = pyuvdata.UVData()
    data_uv.read(data_file)
    for use_lst in list(set(data_uv.lst_array)):
        if use_lst <= np.min(model_lsts):
            use_lst += 2 * np.pi
        elif use_lst >= np.max(model_lsts):
            use_lst -= 2 * np.pi
        lst_distance = np.abs(model_lsts - use_lst)
        ind1 = np.where(lst_distance == np.min(lst_distance))[0]
        ind2 = np.where(lst_distance == np.sort(lst_distance)[1])[0]
        lst1 = model_lsts[ind1]
        model_filename1 = model_lst_filenames[ind1]
        lst2 = model_lsts[ind2]
        model_filename2 = model_lst_filenames[ind2]

        lst_spacing = np.abs(lst2 - lst1)
        lst_spacing1 = np.abs(lst1 - use_lst)
        lst_spacing2 = np.abs(lst2 - use_lst)
        model1_uv = pyuvdata.UVData()
        model1_uv.read(model_filename1)
        model1_uv.data_array *= lst_spacing1 / lst_spacing
        model2_uv = pyuvdata.UVData()
        model2_uv.read(model_filename2)
        model2_uv.data_array *= lst_spacing2 / lst_spacing
        model_uv = model1_uv.sum_vis(model2_uv, inplace=False)
