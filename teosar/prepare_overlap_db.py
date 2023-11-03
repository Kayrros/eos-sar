import numpy as np

from eos.sar.utils import write_array
from teosar import inout, tsinsar
from teosar.utils import OvlIfg


def to_ovl_arr(array, overlap_roi_info, osid):
    return write_array(
        array,
        overlap_roi_info.all_write_rois[osid],
        overlap_roi_info.all_out_shapes[osid],
    )


def get_f_dop(resampler_per_osid, overlap_roi_info_per_swath, osid):
    # also save the doppler frequency at the overlap
    # Now osid corresponds to a single swath
    swath = osid.bsid().split("_")[1].lower()

    resampler = resampler_per_osid[osid]

    eta, ref_time, dop_centroid, dop_rate = resampler.get_doppler_params_gridded(
        np.arange(resampler.dst_shape[0]),
        np.arange(resampler.dst_shape[1]),
        resampler.src_roi_in_burst.get_origin(),
        matrix_to_doppler_frame_roi=resampler.matrix,
    )

    f_dop_array = dop_rate * (eta - ref_time) + dop_centroid

    return to_ovl_arr(f_dop_array, overlap_roi_info_per_swath[swath], osid)


def normalize_cmpx_values(cmpx_values):
    amp = np.abs(cmpx_values)
    non_zero_mask = amp != 0
    normalized = np.copy(cmpx_values)
    normalized[non_zero_mask] = normalized[non_zero_mask] / (amp[non_zero_mask] + 1e-12)
    return normalized


def main(
    dstdir,
    product_ids_per_date,
    orbit_type,
    polarization,
    calibrate,
    get_complex,
    bistatic,
    apd,
    intra_pulse,
    alt_fm_mismatch,
    dem_sampling_ratio,
    primary_id,
    osids_of_interest=None,
):
    pipelines = tsinsar.main_ovl(
        dstdir,
        product_ids_per_date,
        orbit_type,
        polarization,
        calibrate,
        get_complex,
        bistatic,
        apd,
        intra_pulse,
        alt_fm_mismatch,
        dem_sampling_ratio,
        primary_id,
        osids_of_interest=osids_of_interest,
    )

    # TODO all of theses can be should be read from the log, proc and meta files
    dates = [p.date for p in pipelines]
    # the code below works on two dates only for now
    assert len(dates) == 2, "The code below works only on two dates for now"
    dir_builder = inout.OvlDirectoryBuilder(dstdir)
    dir_reader = inout.OvlDirectoryReader(dir_builder)
    primary_pipeline = pipelines[primary_id]
    # TODO change this line in the future
    secondary_pipeline = pipelines[0] if primary_id else pipelines[1]

    resampler_per_osid = primary_pipeline.resampler_per_osid
    overlap_roi_info_per_swath = primary_pipeline.ovl_roi_info_per_swath

    swaths = primary_pipeline.swaths_of_interest

    az_frequency = primary_pipeline.swath_models_per_swath[
        "iw1"
    ].coordinate.azimuth_frequency

    px_shift_per_bsint = {}

    for swath in swaths:
        bsint_of_interest_in_swath = secondary_pipeline.bsint_of_interest_per_swath[
            swath
        ]

        for bsint in bsint_of_interest_in_swath:
            # Do the forward and backward interferograms
            osids = bsint.osids()
            ifg_per_osid = {osid: OvlIfg(dir_reader, *dates, osid) for osid in osids}

            # here we should have only two osids per bsint
            init_ifg = {osid: ifg_per_osid[osid].get_init_interf() for osid in osids}
            ovl_interf = init_ifg[osids[0]] * np.conj(init_ifg[osids[1]])

            f_dop = {
                osid: get_f_dop(resampler_per_osid, overlap_roi_info_per_swath, osid)
                for osid in osids
            }

            delta_f = f_dop[osids[0]] - f_dop[osids[1]]

            K_shift_to_phase = 2 * np.pi * delta_f / az_frequency

            valid_mask = np.logical_not(np.isnan(ovl_interf))

            normalized_vals = normalize_cmpx_values(ovl_interf[valid_mask])

            phi_agg = np.angle(np.mean(normalized_vals))

            px_shift = np.nanmean(phi_agg / (K_shift_to_phase[valid_mask] + 1e-12))

            px_shift_per_bsint[str(bsint)] = px_shift

            for osid in osids:
                esd_correction = np.exp(
                    -1j * 2 * np.pi * f_dop[osid] / az_frequency * px_shift,
                    dtype=np.complex64,
                )

                before_esd = ifg_per_osid[osid].get_topo_corrected()
                inout.save_img(dir_builder.get_ifg_path(osid, *dates), before_esd)

                # px_shift_per_bsint[bsint] = px_shift
                esd_corrected = before_esd * esd_correction

                inout.save_img(
                    dir_builder.get_ifg_path(osid, *dates, esd=True), esd_corrected
                )

    inout.dict_to_json(px_shift_per_bsint, dir_builder.get_ifg_meta_path(*dates))
