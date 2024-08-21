import numpy as np
import pyuvdata
from newcal import calibration_wrappers, calibration_optimization, caldata
import os


def run_abscal():

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
    output_path = "/safepool/rbyrne/hera_abscal"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]

    for file_ind, datafile_name in enumerate(datafile_names):

        print(f"Processing file {file_ind+1} of {len(datafile_names)}")

        data_path = f"{data_filepath}/{datafile_name}.uvh5"
        model_path = f"{model_filepath}/{datafile_name}_model.uvfits"
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
        use_polarizations = [
            pol for pol in model.polarization_array if pol in data.polarization_array
        ]
        data.select(bls=use_baselines, polarizations=use_polarizations)
        model.select(bls=use_baselines, polarizations=use_polarizations)

        # Align phasing
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))

        # null = data.sum_vis(model, inplace=False, override_params=["nsample_array", "earth_omega", "flag_array", "filename", "phase_center_catalog", "timesys"])  # Verify that metadata match

        abscal_params = calibration_wrappers.absolute_calibration(
            data,
            model,
            log_file_path=f"{output_path}/calibration_logs/{datafile_name}.txt",
            verbose=True,
        )

        data = pyuvdata.UVData()
        data.read(data_path)
        data.phase_to_time(np.mean(data.time_array))
        calibration_wrappers.apply_abscal(
            data, abscal_params, data.polarization_array, inplace=True
        )
        data.write_uvfits(f"{output_path}/{datafile_name}_abscal.uvfits")
        with open(f"{output_path}/{datafile_name}_abscal_params.npy", "wb") as f:
            np.save(f, abscal_params)


def run_dw_abscal():

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
    output_path = "/safepool/rbyrne/hera_abscal"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]

    datafile_names = datafile_names[:1]

    avg_spectra = np.load("/safepool/rbyrne/hera_abscal/mean_variance.npz")
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    frequencies = avg_spectra["frequencies"]
    delay_axis = np.fft.fftshift(
        np.fft.fftfreq(
            len(frequencies),
            d=(np.max(frequencies) - np.min(frequencies)) / (len(frequencies) - 1),
        )
    )

    for file_ind, datafile_name in enumerate(datafile_names):

        print(f"Processing file {file_ind+1} of {len(datafile_names)}")

        data_path = f"{data_filepath}/{datafile_name}.uvh5"
        model_path = f"{model_filepath}/{datafile_name}_model.uvfits"
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
        use_polarizations = [
            pol for pol in model.polarization_array if pol in data.polarization_array
        ]
        data.select(bls=use_baselines, polarizations=use_polarizations)
        model.select(bls=use_baselines, polarizations=use_polarizations)

        # Align phasing
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))

        data.compress_by_redundancy()
        model.compress_by_redundancy()

        # null = data.sum_vis(model, inplace=False, override_params=["nsample_array", "earth_omega", "flag_array", "filename", "phase_center_catalog", "timesys"])  # Verify that metadata match

        initial_abscal_params = np.load(
            f"{output_path}/{datafile_name}_abscal_params.npy"
        )
        abscal_params = calibration_wrappers.dw_absolute_calibration(
            data,
            model,
            delay_spectrum_variance,
            bl_length_bin_edges,
            delay_axis,
            # initial_abscal_params=initial_abscal_params,
            log_file_path=f"{output_path}/calibration_logs/{datafile_name}_dwcal.txt",
            verbose=True,
        )

        data = pyuvdata.UVData()
        data.read(data_path)
        data.phase_to_time(np.mean(data.time_array))
        calibration_wrappers.apply_abscal(
            data, abscal_params, data.polarization_array, inplace=True
        )
        data.write_uvfits(f"{output_path}/{datafile_name}_dw_abscal.uvfits")
        with open(f"{output_path}/{datafile_name}_dw_abscal_params.npy", "wb") as f:
            np.save(f, abscal_params)


def run_abscal_May24():

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
    output_path = "/safepool/rbyrne/hera_abscal"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]

    datafile_names = datafile_names[:1]

    for file_ind, datafile_name in enumerate(datafile_names):

        print(f"Processing file {file_ind+1} of {len(datafile_names)}")

        data_path = f"{data_filepath}/{datafile_name}.uvh5"
        model_path = f"{model_filepath}/{datafile_name}_model.uvfits"
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
        use_polarizations = [
            pol for pol in model.polarization_array if pol in data.polarization_array
        ]
        data.select(bls=use_baselines, polarizations=use_polarizations)
        model.select(bls=use_baselines, polarizations=use_polarizations)

        # Align phasing
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))

        data.compress_by_redundancy()
        model.compress_by_redundancy()

        # null = data.sum_vis(model, inplace=False, override_params=["nsample_array", "earth_omega", "flag_array", "filename", "phase_center_catalog", "timesys"])  # Verify that metadata match

        initial_abscal_params = np.load(
            f"{output_path}/{datafile_name}_abscal_params.npy"
        )
        abscal_params = calibration_wrappers.absolute_calibration(
            data,
            model,
            log_file_path=f"{output_path}/calibration_logs/{datafile_name}_abscal_May24.txt",
            verbose=True,
        )

        data = pyuvdata.UVData()
        data.read(data_path)
        data.phase_to_time(np.mean(data.time_array))
        calibration_wrappers.apply_abscal(
            data, abscal_params, data.polarization_array, inplace=True
        )
        data.write_uvfits(f"{output_path}/{datafile_name}_abscal_May24.uvfits")
        with open(f"{output_path}/{datafile_name}_abscal_params_May24.npy", "wb") as f:
            np.save(f, abscal_params)


def run_abscal_Jul8():

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
    output_path = "/safepool/rbyrne/hera_abscal_Jun2024"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]
    datafile_names = datafile_names[:1]
    print(datafile_names)

    for file_ind, datafile_name in enumerate(datafile_names):

        print(f"Processing file {file_ind+1} of {len(datafile_names)}")

        data_path = f"{data_filepath}/{datafile_name}.uvh5"
        model_path = f"{model_filepath}/{datafile_name}_model.uvfits"
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
        use_polarizations = [
            pol for pol in model.polarization_array if pol in data.polarization_array
        ]
        data.select(bls=use_baselines, polarizations=use_polarizations)
        model.select(bls=use_baselines, polarizations=use_polarizations)

        # Align phasing
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))

        data.compress_by_redundancy()
        model.compress_by_redundancy()

        # null = data.sum_vis(model, inplace=False, override_params=["nsample_array", "earth_omega", "flag_array", "filename", "phase_center_catalog", "timesys"])  # Verify that metadata match

        abscal_params = calibration_wrappers.absolute_calibration(
            data,
            model,
            log_file_path=f"{output_path}/calibration_logs/{datafile_name}_abscal_Jul8.txt",
            verbose=True,
        )

        data = pyuvdata.UVData()
        data.read(data_path)
        data.phase_to_time(np.mean(data.time_array))
        calibration_wrappers.apply_abscal(
            data, abscal_params, data.polarization_array, inplace=True
        )
        data.write_uvfits(f"{output_path}/{datafile_name}_abscal.uvfits")
        with open(f"{output_path}/{datafile_name}_abscal_params.npy", "wb") as f:
            np.save(f, abscal_params)


def run_dwabscal_Jul12():

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
    output_path = "/safepool/rbyrne/hera_abscal_Jun2024"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]
    datafile_names = datafile_names[:1]
    print(datafile_names)

    avg_spectra = np.load(
        "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
    )
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    delay_axis = avg_spectra["delay_array"]

    for file_ind, datafile_name in enumerate(datafile_names):

        print(f"Processing file {file_ind+1} of {len(datafile_names)}")

        data_path = f"{data_filepath}/{datafile_name}.uvh5"
        model_path = f"{model_filepath}/{datafile_name}_model.uvfits"
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
        # use_polarizations = [
        #    pol for pol in model.polarization_array if pol in data.polarization_array
        # ]
        use_polarizations = -5
        data.select(bls=use_baselines, polarizations=use_polarizations)
        model.select(bls=use_baselines, polarizations=use_polarizations)

        # Align phasing
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))

        data.compress_by_redundancy()
        model.compress_by_redundancy()

        # null = data.sum_vis(model, inplace=False, override_params=["nsample_array", "earth_omega", "flag_array", "filename", "phase_center_catalog", "timesys"])  # Verify that metadata match

        abscal_params = calibration_wrappers.dw_absolute_calibration(
            data,
            model,
            delay_spectrum_variance,
            bl_length_bin_edges,
            delay_axis,
            log_file_path=f"{output_path}/calibration_logs/{datafile_name}_dwabscal_Powell_Jul12.txt",
            verbose=True,
        )

        data = pyuvdata.UVData()
        data.read(data_path)
        data.select(polarizations=use_polarizations)
        data.phase_to_time(np.mean(data.time_array))
        calibration_wrappers.apply_abscal(
            data, abscal_params, data.polarization_array, inplace=True
        )
        data.write_uvfits(f"{output_path}/{datafile_name}_dwabscal_powell.uvfits")
        with open(
            f"{output_path}/{datafile_name}_dwabscal_powell_params.npy", "wb"
        ) as f:
            np.save(f, abscal_params)


def apply_abscal_solutions_fixed_normalization():

    data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
    abscal_solution_filepath = "/safepool/rbyrne/hera_abscal_Jun2024"
    use_pol_ind = 0

    filenames = os.listdir(abscal_solution_filepath)
    filenames = [name for name in filenames if name.endswith("_dwabscal_params.npy")]
    filenames = [name.removesuffix("_dwabscal_params.npy") for name in filenames]
    filenames = ["zen.2459861.45004.sum.abs_calibrated.red_avg"]

    for file_ind, use_filename in enumerate(filenames):
        dwabscal_params = np.load(
            f"{abscal_solution_filepath}/{use_filename}_dwabscal_params.npy"
        )
        abscal_params = np.load(
            f"{abscal_solution_filepath}/{use_filename}_abscal_params.npy"
        )
        mean_amp_dwabscal = np.nanmean(dwabscal_params[0, :, use_pol_ind])
        mean_amp_abscal = np.nanmean(abscal_params[0, :, use_pol_ind])

        use_abscal_params = np.copy(dwabscal_params)
        use_abscal_params[0, :, use_pol_ind] *= mean_amp_abscal / mean_amp_dwabscal

        print(np.sum(abscal_params[0, :, use_pol_ind]))
        print(np.sum(use_abscal_params[0, :, use_pol_ind]))
        print(abscal_params[0, :, use_pol_ind])
        print(use_abscal_params[0, :, use_pol_ind])
        data = pyuvdata.UVData()
        data.read(f"{data_filepath}/{use_filename}.uvh5")
        data.select(polarizations=-5)
        data.phase_to_time(np.mean(data.time_array))
        data_debug = calibration_wrappers.apply_abscal(
            data, abscal_params, data.polarization_array, inplace=False
        )
        print(np.sum(np.abs(data_debug.data_array[200, 0, :, 0]) ** 2.0))
        calibration_wrappers.apply_abscal(
            data, use_abscal_params, data.polarization_array, inplace=True
        )
        print(np.sum(np.abs(data.data_array[200, 0, :, 0]) ** 2.0))
        data.write_uvfits(
            f"{abscal_solution_filepath}/{use_filename}_dwabscal_normalized.uvfits"
        )


def debug_dwabscal_Jul5():

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

    if False:
        caldata_list = caldata_obj.expand_in_frequency()
        abscal_parameters = np.copy(caldata_obj.abscal_params)
        for freq_ind in range(caldata_obj.Nfreqs):
            calibration_optimization.run_abscal_optimization_single_freq(
                caldata_list[freq_ind],
                1e-6,
                100,
                verbose=True,
            )
            abscal_parameters[:, [freq_ind], :] = caldata_list[freq_ind].abscal_params[
                :, [0], :
            ]

        with open(
            f"/safepool/rbyrne/debug_dwabscal_Jul2024/abscal_params.npy",
            "wb",
        ) as f:
            np.save(f, abscal_parameters)

    dwabscal_parameters = calibration_optimization.run_dw_abscal_optimization(
        caldata_obj,
        1e-6,
        100,
        verbose=True,
    )
    with open(
        f"/safepool/rbyrne/debug_dwabscal_Jul2024/dwabscal_params.npy",
        "wb",
    ) as f:
        np.save(f, dwabscal_parameters)

    caldata_obj.dwcal_inv_covariance[:, :, :, :, :] = 0.0 + 1j * 0.0
    for freq_ind in range(caldata_obj.Nfreqs):
        caldata_obj.dwcal_inv_covariance[:, :, freq_ind, freq_ind, :] = 1.0 + 1j * 0.0
    dwabscal_parameters_diagonal = calibration_optimization.run_dw_abscal_optimization(
        caldata_obj,
        1e-6,
        100,
        verbose=True,
    )
    with open(
        f"/safepool/rbyrne/debug_dwabscal_Jul2024/dwabscal_params_diagonal.npy",
        "wb",
    ) as f:
        np.save(f, dwabscal_parameters_diagonal)


def run_dwabscal_Jul17():

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    data_filepath = "/safepool/rbyrne/hera_data/H6C-data/2459861"
    output_path = "/safepool/rbyrne/hera_abscal_Jun2024"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]
    datafile_names = datafile_names[:1]
    print(datafile_names)

    avg_spectra = np.load(
        "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
    )
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    delay_axis = avg_spectra["delay_array"]

    for file_ind, datafile_name in enumerate(datafile_names):

        print(f"Processing file {file_ind+1} of {len(datafile_names)}")

        data_path = f"{data_filepath}/{datafile_name}.uvh5"
        model_path = f"{model_filepath}/{datafile_name}_model.uvfits"
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
        # use_polarizations = [
        #    pol for pol in model.polarization_array if pol in data.polarization_array
        # ]
        use_polarizations = -5
        data.select(bls=use_baselines, polarizations=use_polarizations)
        model.select(bls=use_baselines, polarizations=use_polarizations)

        # Align phasing
        data.phase_to_time(np.mean(data.time_array))
        model.phase_to_time(np.mean(data.time_array))

        data.compress_by_redundancy()
        model.compress_by_redundancy()

        # null = data.sum_vis(model, inplace=False, override_params=["nsample_array", "earth_omega", "flag_array", "filename", "phase_center_catalog", "timesys"])  # Verify that metadata match

        abscal_params = calibration_wrappers.dw_absolute_calibration(
            data,
            model,
            delay_spectrum_variance,
            bl_length_bin_edges,
            delay_axis,
            log_file_path=f"{output_path}/calibration_logs/{datafile_name}_dwabscal_maxiter2_Jul17.txt",
            verbose=True,
            xtol=1e-4,
            maxiter=2,
        )

        data = pyuvdata.UVData()
        data.read(data_path)
        data.select(polarizations=use_polarizations)
        data.phase_to_time(np.mean(data.time_array))
        calibration_wrappers.apply_abscal(
            data, abscal_params, data.polarization_array, inplace=True
        )
        data.write_uvfits(f"{output_path}/{datafile_name}_dwabscal_maxiter2.uvfits")
        with open(
            f"{output_path}/{datafile_name}_dwabscal_maxiter2_params.npy", "wb"
        ) as f:
            np.save(f, abscal_params)


def run_abscal_Jul24():

    output_path = "/safepool/rbyrne/hera_abscal_Jul2024"

    avg_spectra = np.load(
        "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
    )
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    delay_axis = avg_spectra["delay_array"]

    data_path = "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
    model_path = "/safepool/rbyrne/hera_data/interpolated_models/zen.2459861.45004.sum.abs_calibrated.red_avg_model.uvfits"
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
    use_polarizations = -5  # Use XX only
    data.select(bls=use_baselines, polarizations=use_polarizations)
    model.select(bls=use_baselines, polarizations=use_polarizations)

    # Align phasing
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))

    data.compress_by_redundancy()
    model.compress_by_redundancy()

    # Run abscal
    abscal_params = calibration_wrappers.absolute_calibration(
        data,
        model,
        log_file_path=f"{output_path}/abscal_log.txt",
        verbose=True,
        xtol=1e-7,
        maxiter=50,
    )
    with open(f"{output_path}/abscal_params.npy", "wb") as f:
        np.save(f, abscal_params)

    data = pyuvdata.UVData()
    data.read(data_path)
    data.select(polarizations=use_polarizations)
    data.phase_to_time(np.mean(data.time_array))
    abscal_data = calibration_wrappers.apply_abscal(
        data, abscal_params, data.polarization_array, inplace=False
    )
    abscal_data.write_uvfits(f"{output_path}/abscal_data.uvfits")


def run_dwabscal_Jul24():

    output_path = "/safepool/rbyrne/hera_abscal_Jul2024"

    avg_spectra = np.load(
        "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
    )
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    delay_axis = avg_spectra["delay_array"]

    data_path = "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
    model_path = "/safepool/rbyrne/hera_data/interpolated_models/zen.2459861.45004.sum.abs_calibrated.red_avg_model.uvfits"
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
    use_polarizations = -5  # Use XX only
    data.select(bls=use_baselines, polarizations=use_polarizations)
    model.select(bls=use_baselines, polarizations=use_polarizations)

    # Align phasing
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))

    data.compress_by_redundancy()
    model.compress_by_redundancy()

    # Run DWAbscal
    dwabscal_params = calibration_wrappers.dw_absolute_calibration(
        data,
        model,
        delay_spectrum_variance,
        bl_length_bin_edges,
        delay_axis,
        # initial_abscal_params=abscal_params,
        log_file_path=f"{output_path}/dwabscal_log.txt",
        verbose=True,
        xtol=1e-7,
        maxiter=50,
    )
    with open(f"{output_path}/dwabscal_params.npy", "wb") as f:
        np.save(f, dwabscal_params)

    data = pyuvdata.UVData()
    data.read(data_path)
    data.select(polarizations=use_polarizations)
    data.phase_to_time(np.mean(data.time_array))
    dwabscal_data = calibration_wrappers.apply_abscal(
        data, dwabscal_params, data.polarization_array, inplace=False
    )
    dwabscal_data.write_uvfits(f"{output_path}/dwabscal_data.uvfits")


def run_dwabscal_diagonal_weighting_Jul26():

    output_path = "/safepool/rbyrne/hera_abscal_Jul2024"

    avg_spectra = np.load(
        "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
    )
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    delay_axis = avg_spectra["delay_array"]

    data_path = "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
    model_path = "/safepool/rbyrne/hera_data/interpolated_models/zen.2459861.45004.sum.abs_calibrated.red_avg_model.uvfits"
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
    use_polarizations = -5  # Use XX only
    data.select(bls=use_baselines, polarizations=use_polarizations)
    model.select(bls=use_baselines, polarizations=use_polarizations)

    # Align phasing
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))

    data.compress_by_redundancy()
    model.compress_by_redundancy()

    # Run DWAbscal
    dwabscal_params = calibration_wrappers.dw_absolute_calibration(
        data,
        model,
        np.ones_like(delay_spectrum_variance),
        bl_length_bin_edges,
        delay_axis,
        # initial_abscal_params=abscal_params,
        log_file_path=f"{output_path}/dwabscal_diagonal_log.txt",
        verbose=True,
        xtol=1e-7,
        maxiter=50,
    )
    with open(f"{output_path}/dwabscal_diagonal_params.npy", "wb") as f:
        np.save(f, dwabscal_params)

    data = pyuvdata.UVData()
    data.read(data_path)
    data.select(polarizations=use_polarizations)
    data.phase_to_time(np.mean(data.time_array))
    dwabscal_data = calibration_wrappers.apply_abscal(
        data, dwabscal_params, data.polarization_array, inplace=False
    )
    dwabscal_data.write_uvfits(f"{output_path}/dwabscal_diagonal_data.uvfits")


def run_abscal_Aug8():

    output_path = "/safepool/rbyrne/hera_abscal_Aug2024"

    avg_spectra = np.load(
        "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
    )
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    delay_axis = avg_spectra["delay_array"]

    data_path = "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
    model_path = "/safepool/rbyrne/hera_data/interpolated_models/zen.2459861.45004.sum.abs_calibrated.red_avg_model.uvfits"
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
    use_polarizations = -5  # Use XX only
    data.select(bls=use_baselines, polarizations=use_polarizations)
    model.select(bls=use_baselines, polarizations=use_polarizations)

    # Align phasing
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))

    data.compress_by_redundancy()
    model.compress_by_redundancy()

    # Run abscal
    abscal_params = calibration_wrappers.absolute_calibration(
        data,
        model,
        log_file_path=f"{output_path}/abscal_log.txt",
        verbose=True,
        xtol=1e-7,
        maxiter=50,
    )
    with open(f"{output_path}/abscal_params.npy", "wb") as f:
        np.save(f, abscal_params)

    data = pyuvdata.UVData()
    data.read(data_path)
    data.select(polarizations=use_polarizations)
    data.phase_to_time(np.mean(data.time_array))
    abscal_data = calibration_wrappers.apply_abscal(
        data, abscal_params, data.polarization_array, inplace=False
    )
    abscal_data.write_uvfits(f"{output_path}/abscal_data.uvfits")


def run_dwabscal_Aug8():

    output_path = "/safepool/rbyrne/hera_abscal_Aug2024"

    avg_spectra = np.load(
        "/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz"
    )
    delay_spectrum_variance = avg_spectra["variance"]
    bl_length_bin_edges = avg_spectra["bl_bin_edges"]
    delay_axis = avg_spectra["delay_array"]

    data_path = "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
    model_path = "/safepool/rbyrne/hera_data/interpolated_models/zen.2459861.45004.sum.abs_calibrated.red_avg_model.uvfits"
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
    use_polarizations = -5  # Use XX only
    data.select(bls=use_baselines, polarizations=use_polarizations)
    model.select(bls=use_baselines, polarizations=use_polarizations)

    # Align phasing
    data.phase_to_time(np.mean(data.time_array))
    model.phase_to_time(np.mean(data.time_array))

    data.compress_by_redundancy()
    model.compress_by_redundancy()

    # Run DWAbscal
    dwabscal_params = calibration_wrappers.dw_absolute_calibration(
        data,
        model,
        delay_spectrum_variance,
        bl_length_bin_edges,
        delay_axis,
        # initial_abscal_params=abscal_params,
        log_file_path=f"{output_path}/dwabscal_log.txt",
        verbose=True,
        xtol=1e-7,
        maxiter=50,
    )
    with open(f"{output_path}/dwabscal_params.npy", "wb") as f:
        np.save(f, dwabscal_params)

    data = pyuvdata.UVData()
    data.read(data_path)
    data.select(polarizations=use_polarizations)
    data.phase_to_time(np.mean(data.time_array))
    dwabscal_data = calibration_wrappers.apply_abscal(
        data, dwabscal_params, data.polarization_array, inplace=False
    )
    dwabscal_data.write_uvfits(f"{output_path}/dwabscal_data.uvfits")


if __name__ == "__main__":
    # run_abscal_Aug8()
    run_dwabscal_Aug8()
