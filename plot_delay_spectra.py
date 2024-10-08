import numpy as np
import os
import sys
import pyuvdata
import matplotlib
import matplotlib.pyplot as plt
import sklearn.gaussian_process


def calculate_delay_spectra(uvdata, bl_bin_edges, use_polarization=-5, use_gpr=True):

    use_data = np.copy(uvdata.data_array)
    use_data = use_data[
        :, :, np.where(uvdata.polarization_array == use_polarization)[0][0]
    ]
    use_flags = np.copy(uvdata.flag_array)
    use_flags = use_flags[
        :, :, np.where(uvdata.polarization_array == use_polarization)[0][0]
    ]
    baseline_all_flagged = np.min(
        use_flags, axis=(1,)
    )  # Note what baselines are fully flagged
    if use_gpr:
        for bl_ind in range(uvdata.Nblts):
            if baseline_all_flagged[bl_ind]:
                use_data[bl_ind, :] = np.nan + 1j * np.nan
            else:
                bl_data = use_data[bl_ind, :]
                bl_flags = use_flags[bl_ind, :]
                if np.max(bl_flags):  # Fill in flagged data
                    gp_real = sklearn.gaussian_process.GaussianProcessRegressor(
                        normalize_y=True
                    )
                    gp_real.fit(
                        uvdata.freq_array[np.where(~bl_flags)].reshape(-1, 1),
                        np.real(bl_data[np.where(~bl_flags)]),
                    )
                    gpr_values_real = gp_real.predict(
                        uvdata.freq_array[np.where(bl_flags)].reshape(-1, 1)
                    )
                    gp_imag = sklearn.gaussian_process.GaussianProcessRegressor(
                        normalize_y=True
                    )
                    gp_imag.fit(
                        uvdata.freq_array[np.where(~bl_flags)].reshape(-1, 1),
                        np.imag(bl_data[np.where(~bl_flags)]),
                    )
                    gpr_values_imag = gp_imag.predict(
                        uvdata.freq_array[np.where(bl_flags)].reshape(-1, 1)
                    )
                    bl_data[np.where(bl_flags)] = gpr_values_real + 1j * gpr_values_imag
                use_data[bl_ind, :] = bl_data
    else:  # Zero out flagged data
        use_data[np.where(use_flags)] = 0  # Zero out flagged data

    # FFT across frequency
    delay_array = np.fft.fftfreq(uvdata.Nfreqs, d=uvdata.channel_width)
    delay_array = np.fft.fftshift(delay_array)
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft(use_data, axis=1), axes=1))
    fft_abs *= np.mean(uvdata.channel_width)

    # Average in baseline length bins
    bl_lengths = np.sqrt(np.sum(uvdata.uvw_array**2.0, axis=1))
    binned_variance = np.full(
        [len(bl_bin_edges) - 1, uvdata.Nfreqs], np.nan, dtype="float"
    )
    nsamples = np.full([len(bl_bin_edges) - 1, uvdata.Nfreqs], 0.0, dtype="float")
    for bin_ind in range(len(bl_bin_edges) - 1):
        bl_inds = np.where(
            (bl_lengths > bl_bin_edges[bin_ind])
            & (bl_lengths <= bl_bin_edges[bin_ind + 1])
            & (~baseline_all_flagged)
        )[0]
        if len(bl_inds) > 0:
            binned_variance[bin_ind, :] = np.nansum(fft_abs[bl_inds, :] ** 2.0, axis=0)
            nsamples[bin_ind, :] = len(bl_inds)

    return binned_variance / nsamples


def plot_visibilities(
    uvdata,
    savepath=None,
    plot_horizon_lines=False,
    use_polarization=-5,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    nbins=50,
    vmin=1e16,
    vmax=1e22,
    use_gpr=True,
):

    bl_lengths = np.sqrt(np.sum(uvdata.uvw_array**2.0, axis=1))
    delay_array = np.fft.fftfreq(uvdata.Nfreqs, d=np.mean(uvdata.channel_width))
    if xmin is None:
        xmin = np.min(bl_lengths)
    if xmax is None:
        xmax = np.max(bl_lengths)
    bl_bin_edges = np.linspace(xmin, xmax, num=nbins + 1)
    binned_variance = calculate_delay_spectra(
        uvdata,
        bl_bin_edges,
        use_polarization=use_polarization,
        use_gpr=use_gpr,
    )

    # Plot
    use_cmap = matplotlib.cm.get_cmap("inferno")
    use_cmap.set_bad(color="whitesmoke")
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    if ymin is None:
        ymin = np.min(delay_array) * 1e6
    if ymax is None:
        ymax = np.max(delay_array) * 1e6
    plt.imshow(
        binned_variance.T,
        origin="lower",
        interpolation="none",
        cmap=use_cmap,
        norm=norm,
        extent=[
            xmin,
            xmax,
            ymin,
            ymax,
        ],
        aspect="auto",
    )

    if plot_horizon_lines:
        plt.plot(
            [np.min(bl_bin_edges), np.max(bl_bin_edges)],
            [
                np.min(bl_bin_edges) / 3e8 * 1e6,
                np.max(bl_bin_edges) / 3e8 * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )
        plt.plot(
            [np.min(bl_bin_edges), np.max(bl_bin_edges)],
            [
                -np.min(bl_bin_edges) / 3e8 * 1e6,
                -np.max(bl_bin_edges) / 3e8 * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )

    cbar = plt.colorbar(extend="both")
    cbar.ax.set_ylabel(
        "Visibility Variance (Jy$^{2}$/s$^2$)", rotation=270, labelpad=15
    )
    plt.xlabel("Baseline Length (m)")
    plt.ylim([np.min(delay_array) * 1e6, np.max(delay_array) * 1e6])
    plt.ylabel("Delay ($\mu$s)")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
        plt.close()


def plot_difference(
    uvdata1,
    uvdata2,
    savepath=None,
    plot_horizon_lines=False,
    use_polarization=-5,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    nbins=50,
    extent=1e22,
    ratio=False,
):

    bl_lengths = np.sqrt(np.sum(uvdata1.uvw_array**2.0, axis=1))
    delay_array = np.fft.fftfreq(uvdata1.Nfreqs, d=np.mean(uvdata1.channel_width))
    if xmin is None:
        xmin = np.min(bl_lengths)
    if xmax is None:
        xmax = np.max(bl_lengths)
    bl_bin_edges = np.linspace(xmin, xmax, num=nbins + 1)
    binned_variance1 = calculate_delay_spectra(
        uvdata1, bl_bin_edges, use_polarization=use_polarization
    )
    binned_variance2 = calculate_delay_spectra(
        uvdata2, bl_bin_edges, use_polarization=use_polarization
    )

    # Plot
    use_cmap = matplotlib.cm.get_cmap("seismic")
    use_cmap.set_bad(color="whitesmoke")

    if ratio:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    else:
        norm = matplotlib.colors.SymLogNorm(extent / 1e10, vmax=extent)

    if ymin is None:
        ymin = np.min(delay_array) * 1e6
    if ymax is None:
        ymax = np.max(delay_array) * 1e6
    plot_values = binned_variance1.T - binned_variance2.T
    if ratio:
        plot_values /= binned_variance2.T
    plt.imshow(
        plot_values,
        origin="lower",
        interpolation="none",
        cmap=use_cmap,
        norm=norm,
        extent=[
            xmin,
            xmax,
            ymin,
            ymax,
        ],
        aspect="auto",
    )

    if plot_horizon_lines:
        plt.plot(
            [np.min(bl_bin_edges), np.max(bl_bin_edges)],
            [
                np.min(bl_bin_edges) / 3e8 * 1e6,
                np.max(bl_bin_edges) / 3e8 * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )
        plt.plot(
            [np.min(bl_bin_edges), np.max(bl_bin_edges)],
            [
                -np.min(bl_bin_edges) / 3e8 * 1e6,
                -np.max(bl_bin_edges) / 3e8 * 1e6,
            ],
            "--",
            color="white",
            linewidth=1.0,
        )

    cbar = plt.colorbar(extend="both")
    cbar.ax.set_ylabel(
        "Visibility Variance (Jy$^{2}$/s$^2$)", rotation=270, labelpad=15
    )
    plt.xlabel("Baseline Length (m)")
    plt.ylim([np.min(delay_array) * 1e6, np.max(delay_array) * 1e6])
    plt.ylabel("Delay ($\mu$s)")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
        plt.close()


def run_plot_data(
    model_filepath="/safepool/rbyrne/hera_data/interpolated_models",
    calibrated_data_path="/safepool/rbyrne/hera_abscal_Jun2024",
    model_suffix="_model.uvfits",
    calibrated_data_suffix="_abscal.uvfits",
    plot_save_path="/safepool/rbyrne/hera_abscal_Jun2024/delay_spectrum_plots",
):

    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix(model_suffix) for name in model_filenames]

    calibrated_data_path = "/safepool/rbyrne/hera_abscal_Jun2024"
    calibrated_data_filenames = [
        f"{datafile_name}{calibrated_data_suffix}" for datafile_name in datafile_names
    ]

    for file_ind in range(len(datafile_names)):

        # Plot model
        model = pyuvdata.UVData()
        model.read(f"{model_filepath}/{model_filenames[file_ind]}")
        model.inflate_by_redundancy(use_grid_alg=True)
        bl_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
        xmax = np.max(bl_lengths)
        plot_visibilities(
            model,
            savepath=f"{plot_save_path}/{datafile_names[file_ind]}_model.png",
            xmin=0,
            xmax=xmax,
        )

        # Plot data
        data = pyuvdata.UVData()
        data.read(f"{calibrated_data_path}/{calibrated_data_filenames[file_ind]}")
        data.inflate_by_redundancy(use_grid_alg=True)
        plot_visibilities(
            data,
            f"{plot_save_path}/{datafile_names[file_ind]}_data.png",
            xmin=0,
            xmax=xmax,
        )

        # Calculate and plot difference
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
                "telescope_location",
                "vis_units",
                "antenna_names",
                "antenna_positions",
                "instrument",
                "x_orientation",
                "antenna_numbers",
            ],
        )
        diff.flag_array = data.flag_array
        plot_visibilities(
            diff,
            f"{plot_save_path}/{datafile_names[file_ind]}_diff.png",
            xmin=0,
            xmax=xmax,
        )


def calculate_avg_model_error(
    model_filepath="/safepool/rbyrne/hera_data/interpolated_models",
    calibrated_data_path="/safepool/rbyrne/hera_abscal_Jun2024",
    model_suffix="_model.uvfits",
    calibrated_data_suffix="_dwabscal_normalized.uvfits",
    output_file="/safepool/rbyrne/hera_abscal_Jun2024/mean_variance_abscal_nbins200_xx.npz",
    excluded_obs=["zen.2459861.46302", "zen.2459861.47152"],
    use_pol=-5,
    nbins=200,
):

    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix(model_suffix) for name in model_filenames]

    for obs in excluded_obs:
        datafile_names.remove(f"{obs}.sum.abs_calibrated.red_avg")

    calibrated_data_filenames = [
        f"{datafile_name}{calibrated_data_suffix}" for datafile_name in datafile_names
    ]
    print(calibrated_data_filenames)

    for file_ind in range(len(datafile_names)):
        print(f"Processing file {file_ind+1} of {len(datafile_names)}.")

        model = pyuvdata.UVData()
        model.read(f"{model_filepath}/{model_filenames[file_ind]}")
        model.inflate_by_redundancy(use_grid_alg=True)
        bl_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))

        data = pyuvdata.UVData()
        data.read(f"{calibrated_data_path}/{calibrated_data_filenames[file_ind]}")
        data.inflate_by_redundancy(use_grid_alg=True)
        model_baselines = list(set(list(zip(model.ant_1_array, model.ant_2_array))))
        data_baselines = list(set(list(zip(data.ant_1_array, data.ant_2_array))))
        use_baselines = [
            baseline
            for baseline in model_baselines
            if (baseline in data_baselines) or (baseline[::-1] in data_baselines)
        ]
        data.select(bls=use_baselines, polarizations=use_pol)
        model.select(bls=use_baselines, polarizations=use_pol)
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
            ],
        )
        diff.flag_array = data.flag_array
        bl_lengths = np.sqrt(np.sum(diff.uvw_array**2.0, axis=1))
        use_data = np.copy(diff.data_array[:, :, 0])
        use_data[np.where(diff.flag_array[:, :, 0])] = 0  # Zero out flagged data
        baseline_all_flagged = np.min(
            diff.flag_array, axis=(1, 2, 3)
        )  # Note what baselines are fully flagged

        if file_ind == 0:
            channel_width = diff.channel_width
            frequencies = diff.freq_array.flatten()
            delay_array = np.fft.fftshift(np.fft.fftfreq(diff.Nfreqs, d=channel_width))
            bl_bin_edges = np.linspace(0, np.max(bl_lengths), num=nbins + 1)
            binned_variance = np.full([nbins, len(frequencies)], 0.0, dtype="float")
            nsamples = np.full([nbins, len(frequencies)], 0.0, dtype="float")

        # FFT across frequency
        if diff.channel_width != channel_width:
            print("ERROR: Channel width mismatch. Exiting.")
            sys.exit(1)
        if np.max(np.abs(diff.freq_array.flatten() - frequencies)) != 0.0:
            print("ERROR: Frequency array mismatch. Exiting.")
            sys.exit(1)
        fft_abs = np.abs(np.fft.fftshift(np.fft.fft(use_data, axis=1), axes=1))
        fft_abs *= channel_width

        for bin_ind in range(nbins):
            bl_inds = np.where(
                (bl_lengths > bl_bin_edges[bin_ind])
                & (bl_lengths <= bl_bin_edges[bin_ind + 1])
                & (~baseline_all_flagged)
            )[0]
            if len(bl_inds) > 0:
                binned_variance[bin_ind, :] += np.sum(
                    fft_abs[bl_inds, :] ** 2.0, axis=0
                )
                nsamples[bin_ind, :] += len(bl_inds)

    mean_variance = {}
    mean_variance["variance"] = binned_variance / nsamples
    mean_variance["nsamples"] = nsamples
    mean_variance["delay_array"] = delay_array
    mean_variance["bl_bin_edges"] = bl_bin_edges
    np.savez(output_file, **mean_variance)


if __name__ == "__main__":
    calculate_avg_model_error()
