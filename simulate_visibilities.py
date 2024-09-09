import sys

sys.path.append("/home/rbyrne/rlb_LWA/LWA_data_preprocessing")
from generate_model_vis_fftvis import run_fftvis_diffuse_sim

catalog_path = "/home/rbyrne/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
beam_path = "/safepool/rbyrne/hera_data/beam_models/HERA_NF_efield.beamfits"  # downloaded from https://github.com/HERA-Team/hera_pspec/blob/main/hera_pspec/data/HERA_NF_efield.beamfits
input_data_path = "/safepool/rbyrne/hera_data/H6C-data/2459861/zen.2459861.45004.sum.abs_calibrated.red_avg.uvh5"
output_uvfits_path = (
    "/safepool/rbyrne/hera_data/simulated_model_vis/zen.2459861.45004.fftvis_sim.uvfits"
)
run_fftvis_diffuse_sim(
    map_path=catalog_path,
    beam_path=beam_path,
    input_data_path=input_data_path,
    output_uvfits_path=output_uvfits_path,
)
