import numpy as np
import os
import sys
import pyuvdata
import matplotlib
import matplotlib.pyplot as plt


def plot_visibilities(
    uvdata,
    savepath,
    plot_horizon_lines=False,
    use_polarization=-5,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    nbins=50,
    vmin=1e16,
    vmax=1e22,
):

    use_data = np.copy(uvdata.data_array)
    use_data[np.where(uvdata.flag_array)] = 0  # Zero out flagged data
    use_data = use_data[
        :, :, :, np.where(uvdata.polarization_array == use_polarization)[0]
    ]

    # FFT across frequency
    delay_array = np.fft.fftfreq(uvdata.Nfreqs, d=uvdata.channel_width)
    delay_array = np.fft.fftshift(delay_array)
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft(use_data, axis=2), axes=2))
    fft_abs *= uvdata.channel_width

    # Average in baseline length bins
    bl_lengths = np.sqrt(np.sum(uvdata.uvw_array**2.0, axis=1))
    if xmin is None:
        xmin = np.min(bl_lengths)
    if xmax is None:
        xmax = np.max(bl_lengths)
    bl_bin_edges = np.linspace(xmin, xmax, num=nbins + 1)
    binned_variance = np.full([nbins, uvdata.Nfreqs], np.nan, dtype="float")
    for bin_ind in range(nbins):
        bl_inds = np.where(
            (bl_lengths > bl_bin_edges[bin_ind])
            & (bl_lengths <= bl_bin_edges[bin_ind + 1])
        )[0]
        if len(bl_inds) > 0:
            binned_variance[bin_ind, :] = np.mean(
                fft_abs[bl_inds, 0, :, 0] ** 2.0, axis=0
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
    plt.savefig(savepath, dpi=300)
    plt.close()


def run_plot_data():

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]

    calibrated_data_path = "/safepool/rbyrne/hera_abscal"
    calibrated_data_filenames = [
        f"{datafile_name}_abscal.uvfits" for datafile_name in datafile_names
    ]

    plot_save_path = "/safepool/rbyrne/hera_abscal/delay_spectrum_plots"

    for file_ind in range(len(datafile_names)):

        # Plot model
        model = pyuvdata.UVData()
        model.read(f"{model_filepath}/{model_filenames[file_ind]}")
        model.inflate_by_redundancy(use_grid_alg=True)
        bl_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
        xmax = np.max(bl_lengths)
        plot_visibilities(
            model,
            f"{plot_save_path}/{datafile_names[file_ind]}_model.png",
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


def calculate_avg_model_error():

    output_file = "/safepool/rbyrne/hera_abscal/mean_variance.npz"

    model_filepath = "/safepool/rbyrne/hera_data/interpolated_models"
    model_filenames = os.listdir(model_filepath)
    datafile_names = [name.removesuffix("_model.uvfits") for name in model_filenames]

    excluded_obs = ["zen.2459861.46302", "zen.2459861.47152"]
    for obs in excluded_obs:
        datafile_names.remove(f"{obs}.sum.abs_calibrated.red_avg")

    calibrated_data_path = "/safepool/rbyrne/hera_abscal"
    calibrated_data_filenames = [
        f"{datafile_name}_abscal.uvfits" for datafile_name in datafile_names
    ]

    for file_ind in range(len(datafile_names)):
        print(f"Processing file {file_ind+1} of {len(datafile_names)}.")

        # Plot model
        model = pyuvdata.UVData()
        model.read(f"{model_filepath}/{model_filenames[file_ind]}")
        model.inflate_by_redundancy(use_grid_alg=True)
        bl_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
        xmax = np.max(bl_lengths)

        # Plot data
        data = pyuvdata.UVData()
        data.read(f"{calibrated_data_path}/{calibrated_data_filenames[file_ind]}")
        data.inflate_by_redundancy(use_grid_alg=True)

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
                "phase_center_frame_pa",
                "telescope_location",
                "vis_units",
                "antenna_names",
                "antenna_positions",
                "instrument",
                "x_orientation",
                "antenna_numbers",
                "lst_array",
                "time_array",
            ],
        )
        diff.flag_array = data.flag_array
        bl_lengths = np.sqrt(np.sum(diff.uvw_array**2.0, axis=1))

        if file_ind == 0:
            channel_width = diff.channel_width
            frequencies = diff.freq_array.flatten()
            delay_array = np.fft.fftshift(np.fft.fftfreq(diff.Nfreqs, d=channel_width))
            nbins = 200
            bl_bin_edges = np.linspace(0, np.max(bl_lengths), num=nbins + 1)
            binned_variance = np.full([nbins, len(frequencies)], 0.0, dtype="float")
            nsamples = np.full([nbins, len(frequencies)], 0.0, dtype="float")

        use_data = np.copy(diff.data_array)
        use_data[np.where(diff.flag_array)] = 0  # Zero out flagged data
        use_data = use_data[
            :,
            :,
            :,
            np.where(diff.polarization_array == -5)[0],  # Use only XX visibilities
        ]

        # FFT across frequency
        if diff.channel_width != channel_width:
            print("ERROR: Channel width mismatch. Exiting.")
            sys.exit(1)
        if np.max(np.abs(diff.freq_array.flatten() - frequencies)) != 0.0:
            print("ERROR: Frequency array mismatch. Exiting.")
            sys.exit(1)
        fft_abs = np.abs(np.fft.fftshift(np.fft.fft(use_data, axis=2), axes=2))
        fft_abs *= channel_width

        for bin_ind in range(nbins):
            bl_inds = np.where(
                (bl_lengths > bl_bin_edges[bin_ind])
                & (bl_lengths <= bl_bin_edges[bin_ind + 1])
            )[0]
            if len(bl_inds) > 0:
                binned_variance[bin_ind, :] += np.mean(
                    fft_abs[bl_inds, 0, :, 0] ** 2.0, axis=0
                )
                nsamples[bin_ind, :] += len(bl_inds)

    mean_variance = {}
    mean_variance["variance"] = binned_variance / nsamples
    mean_variance["nsamples"] = nsamples
    mean_variance["frequencies"] = frequencies
    mean_variance["bl_bin_edges"] = bl_bin_edges
    np.savez(output_file, **mean_variance)


if __name__ == "__main__":
    calculate_avg_model_error()
