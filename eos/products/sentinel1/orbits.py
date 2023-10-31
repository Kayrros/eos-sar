import datetime
import functools
import glob
import io
import os
import warnings
from typing import Any, Optional, Sequence, Union

from lxml import etree

from eos.sar.orbit import StateVector

from .metadata import (
    Sentinel1BurstMetadata,
    Sentinel1GRDMetadata,
    isostring_to_timestamp,
)


def _string_to_timestamp(s):
    """Convert a string representing a date and time to a float number."""
    return (
        datetime.datetime.strptime(s, "%Y%m%dT%H%M%S")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def _parse_start_end_date_from_orbit_file(s):
    """
    Extract start and end dates for an orbit file filename.

    Args:
      s (str): filename string, formatted as \
      S1A_OPER_AUX_POEORB_OPOD_20161102T122427_V20161012T225943_20161014T005943.EOF

    Return:
      start, end (str): two dates as string (20161012T225943 and 20161014T005943 in the example)
    """
    start = s.split("_")[6][1:]
    end = s.split("_")[7].split(".")[0]
    return start, end


def select_orbit_files_from_filelist(files, date, missionid):
    date = _string_to_timestamp(date)
    missionid = missionid.lower()

    candidates = []
    for file in files:
        filename = os.path.basename(file)

        if filename[: len(missionid)].lower() != missionid:
            continue

        s, e = _parse_start_end_date_from_orbit_file(filename)
        s = _string_to_timestamp(s)
        e = _string_to_timestamp(e)

        # time buffer of 10 state vectors with 10 seconds per state vector before the date
        buffer_pre = 10 * 10
        # time buffer of 20 state vectors with 10 seconds per state vector after the date, since the date often indicates the beginning of the product
        buffer_post = 10 * 10

        if s + buffer_pre < date and e - buffer_post > date:
            candidates.append(file)

    if candidates:
        return sorted(candidates)

    raise FileNotFoundError(
        f"could not find an orbit file for date={date} mission={missionid}"
    )


def retrieve_new_statevectors_to_slc_burst(
    xml_content, burst: Sentinel1BurstMetadata
) -> list[StateVector]:
    return retrieve_new_statevectors_to_slc_bursts(xml_content, [burst])


def retrieve_new_statevectors_to_slc_bursts(
    xml_content: Union[str, bytes, io.BytesIO], bursts: Sequence[Sentinel1BurstMetadata]
) -> list[StateVector]:
    return get_new_list_of_statevectors(xml_content, [b.state_vectors for b in bursts])


def retrieve_new_statevectors_to_grd_meta(
    xml_content: Union[str, bytes, io.BytesIO], meta: Sentinel1GRDMetadata
) -> list[StateVector]:
    return get_new_list_of_statevectors(xml_content, (meta.state_vectors,))


def retrieve_new_statevectors_to_dict_meta(
    xml_content: Union[str, bytes, io.BytesIO], meta: dict[str, Any]
) -> list[StateVector]:
    return get_new_list_of_statevectors(xml_content, (meta["state_vectors"],))


def get_new_list_of_statevectors(
    xml_content: Union[str, bytes, io.BytesIO],
    statevectors_list: Sequence[Sequence[StateVector]],
) -> list[StateVector]:
    # compute the approximative middle time of the burst/product
    # we will extract all orbit data over a window of 3 minutes centered around this middle
    start = min([state_vectors[0].time for state_vectors in statevectors_list])
    end = max([state_vectors[-1].time for state_vectors in statevectors_list])
    mid = (start + end) / 2

    newsvs: list[StateVector] = []

    if isinstance(xml_content, str):
        xml_content = io.BytesIO(xml_content.encode("utf-8"))

    context = etree.iterparse(xml_content, events=("end",), tag="OSV")
    for _, element in context:
        date = isostring_to_timestamp(element.findtext("UTC")[4:])

        if date < mid - 90:
            continue
        if date > mid + 90:
            break

        x = float(element.findtext("X"))
        y = float(element.findtext("Y"))
        z = float(element.findtext("Z"))
        vx = float(element.findtext("VX"))
        vy = float(element.findtext("VY"))
        vz = float(element.findtext("VZ"))
        newsvs.append(
            StateVector(
                time=date,
                position=(x, y, z),
                velocity=(vx, vy, vz),
            )
        )

    return newsvs


def search_valid_orbit_files_from_local_folder(path, product_info, type):
    date, missionid = product_info

    files = glob.glob(
        f"{path}/{missionid.upper()}_OPER_AUX_{type.upper()}ORB_OPOD_*.EOF"
    )
    try:
        files = select_orbit_files_from_filelist(files, date, missionid)
    except FileNotFoundError:
        return None

    return files[-1]


def _retrieve_statevectors_from_source(
    product_info, burst, *, force_type, source
) -> tuple[list[StateVector], str]:
    if isinstance(product_info, tuple):
        # ('20210216T151206', 'S1A')
        date, missionid = product_info
    else:
        # 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
        assert isinstance(product_info, str)
        missionid = product_info[:3]
        date = product_info[17:32]

    def try_for_orbit_type(type) -> Optional[tuple[list[StateVector], str]]:
        xml = source(date, missionid, type)
        if not xml:
            return None

        orbtype = f"orb{type}"
        if isinstance(burst, Sentinel1BurstMetadata):
            statevectors = retrieve_new_statevectors_to_slc_burst(xml, burst)
        elif isinstance(burst, list) and isinstance(burst[0], Sentinel1BurstMetadata):
            statevectors = retrieve_new_statevectors_to_slc_bursts(xml, burst)
        elif isinstance(burst, Sentinel1GRDMetadata):
            statevectors = retrieve_new_statevectors_to_grd_meta(xml, burst)
        elif isinstance(burst, dict):
            statevectors = retrieve_new_statevectors_to_dict_meta(xml, burst)
        else:
            assert False, burst

        return statevectors, orbtype

    if force_type:
        ret = try_for_orbit_type(force_type.replace("orb", ""))
    else:
        ret = try_for_orbit_type("poe") or try_for_orbit_type("res")
    if ret is not None:
        return ret

    raise FileNotFoundError(
        f"could not find an orbit file for date={date} mission={missionid}"
    )


def retrieve_statevectors_using_local_folder(
    path, product_info, burst, b, force_type=None
) -> tuple[list[StateVector], str]:
    """Retrieve the orbit statevectors of the given bursts using a local folder.

    Args
        path: filesystem path to a folder containing .EOF files
        product_info: can be either a S1 SLC product_id (str) or a tuple containing the missionid (str) and the date (str)
        burst: can be either a single burst metadata (Sentinel1BurstMetadata or Sentinel1GRDMetadata) or a list of burst metadata (list[Sentinel1BurstMetadata]). It should be metadata corresponding to a single acquisition/datatake.
        force_type (str, optional): request a specific type of orbit file (can be 'orbres' or 'orbpoe')

    Returns
        The two return values can be used for the *Metadata.with_new_state_vectors() method:
        str: the type of orbit found ('orbres' or 'orbpre')
        list: list of StateVector retrieved

    Raises
        FileNotFoundError: if no orbit file is found for the product_info
    """
    warnings.warn(
        "the sentinel1.orbits module is deprecated, use sentinel1.orbit_catalog instead.",
        DeprecationWarning,
    )

    def source(date, missionid, type):
        file = search_valid_orbit_files_from_local_folder(path, (date, missionid), type)
        if not file:
            return None

        return open(file, "rb")

    return _retrieve_statevectors_from_source(
        product_info, burst, force_type=force_type, source=source
    )


def retrieve_statevectors_using_phoenix(
    phx_client,
    product_info,
    burst,
    *,
    force_type=None,
    phx_source="aws:proxima:kayrros-prod-sentinel-aux",
) -> tuple[list[StateVector], str]:
    """Retrieve the orbit statevectors of the given bursts using the Phoenix catalog.

    Args
        phx_client: phoenix client
        product_info: can be either a S1 SLC product_id (str) or a tuple containing the missionid (str) and the date (str)
        burst: can be either a single burst metadata (Sentinel1BurstMetadata or Sentinel1GRDMetadata) or a list of burst metadata (list[Sentinel1BurstMetadata]). It should be metadata corresponding to a single acquisition/datatake.
        force_type (str, optional): request a specific type of orbit file (can be 'orbres' or 'orbpoe')
        phx_source (str, default to Kayrros Proxima): phoenix source from the esa-sentinel-1-csar-aux collection

    Returns
        The two return values can be used for the *Metadata.with_new_state_vectors() method:
        list: list of StateVector retrieved
        str: the type of orbit found ('orbres' or 'orbpre')

    Raises
        FileNotFoundError: if no orbit file is found for the product_info
    """
    warnings.warn(
        "the sentinel1.orbits module is deprecated, use sentinel1.orbit_catalog instead.",
        DeprecationWarning,
    )

    import phoenix.catalog

    def search_valid_orbit_files_from_phoenix(date, missionid, orbtype):
        col = phx_client.get_collection("esa-sentinel-1-csar-aux").at(phx_source)

        platform = f"sentinel-{missionid[1:].lower()}"
        import dateutil.parser

        date_ = dateutil.parser.parse(date)

        filters = [
            phoenix.catalog.Field("sentinel1:begin_position") < date_,
            phoenix.catalog.Field("sentinel1:end_position") > date_,
            phoenix.catalog.Field("platform") == platform,
            phoenix.catalog.Field("sentinel1:product_type") == f"{orbtype.upper()}ORB",
        ]

        items = col.list_items(filters)

        id_to_items = {it.id: it for it in items}
        ids = list(id_to_items.keys())
        try:
            valid_ids = select_orbit_files_from_filelist(ids, date, missionid)
        except FileNotFoundError:
            return None

        return id_to_items[valid_ids[-1]]

    @functools.lru_cache
    def source(date, missionid, type):
        item = search_valid_orbit_files_from_phoenix(date, missionid, type)
        if not item:
            return None

        xml = io.BytesIO(item.assets.download_as_bytes("PRODUCT"))
        return xml

    return _retrieve_statevectors_from_source(
        product_info, burst, force_type=force_type, source=source
    )
