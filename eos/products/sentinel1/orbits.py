import os
import io
import glob
import datetime
import functools

from lxml import etree

from .metadata import isostring_to_timestamp


def _string_to_timestamp(s):
    """Convert a string representing a date and time to a float number."""
    return datetime.datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=datetime.timezone.utc).timestamp()


def _parse_start_end_date_from_orbit_file(s):
    """
    Extract start and end dates for an orbit file filename.

    Args:
      s (str): filename string, formatted as \
      S1A_OPER_AUX_POEORB_OPOD_20161102T122427_V20161012T225943_20161014T005943.EOF

    Return:
      start, end (str): two dates as string (20161012T225943 and 20161014T005943 in the example)
    """
    start = s.split('_')[6][1:]
    end = s.split('_')[7].split('.')[0]
    return start, end


def select_orbit_files_from_filelist(files, date, missionid):
    date = _string_to_timestamp(date)
    missionid = missionid.lower()

    candidates = []
    for file in files:
        filename = os.path.basename(file)

        if filename[:len(missionid)].lower() != missionid:
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

    raise FileNotFoundError(f'could not find an orbit file for date={date} mission={missionid}')


def apply_new_statevectors_to_burst(xml_content, burst, orbtype):
    apply_new_statevectors_to_bursts(xml_content, [burst], orbtype)


def apply_new_statevectors_to_bursts(xml_content, bursts, orbtype):
    # compute the approximative middle time of the burst/product
    # we will extract all orbit data over a window of 3 minutes centered around this middle
    start = min([burst['state_vectors'][0]['time'] for burst in bursts])
    end = max([burst['state_vectors'][-1]['time'] for burst in bursts])
    mid = (start + end) / 2

    newsvs = [[] for _ in bursts]

    if type(xml_content) == str:
        xml_content = io.BytesIO(xml_content.encode('utf-8'))

    context = etree.iterparse(xml_content, events=('end',), tag='OSV')
    for _, element in context:
        date = isostring_to_timestamp(element.findtext('UTC')[4:])

        if date < mid - 90:
            continue
        if date > mid + 90:
            break

        for i, b in enumerate(bursts):
            newsvs[i].append({
                'time': date,
                'position': [float(element.findtext(k)) for k in ['X', 'Y', 'Z']],
                'velocity': [float(element.findtext(k)) for k in ['VX', 'VY', 'VZ']]
            })

    for i, b in enumerate(bursts):
        # make sure we fetched enough state_vectors
        assert len(newsvs[i]) >= 15
        b['state_vectors'] = newsvs[i]
        b['state_vectors_origin'] = orbtype


def search_valid_orbit_files_from_local_folder(path, product_info, type):
    date, missionid = product_info

    files = glob.glob(f'{path}/{missionid.upper()}_OPER_AUX_{type.upper()}ORB_OPOD_*.EOF')
    try:
        files = select_orbit_files_from_filelist(files, date, missionid)
    except FileNotFoundError:
        return None

    return files[-1]


def _update_statevectors_from_source(product_info, burst, *, force_type, source):
    if isinstance(product_info, tuple):
        # ('20210216T151206', 'S1A')
        date, missionid = product_info
    else:
        # 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
        assert isinstance(product_info, str)
        missionid = product_info[:3]
        date = product_info[17:32]

    def try_for_orbit_type(type):
        xml = source(date, missionid, type)
        if not xml:
            return None

        orbtype = f'orb{type}'
        if isinstance(burst, dict):
            apply_new_statevectors_to_burst(xml, burst, orbtype)
        else:
            apply_new_statevectors_to_bursts(xml, burst, orbtype)

        return orbtype

    if force_type:
        type = try_for_orbit_type(force_type.replace('orb', ''))
    else:
        type = try_for_orbit_type('poe') or try_for_orbit_type('res')
    if type:
        return type

    raise FileNotFoundError(f'could not find an orbit file for date={date} mission={missionid}')


def update_statevectors_using_local_folder(path, product_info, burst, *, force_type=None):
    '''Retrieve the orbit statevectors of the given bursts using a local folder.

    Args
        path: filesystem path to a folder containing .EOF files
        product_info: can be either a S1 SLC product_id (str) or a tuple containing the missionid (str) and the date (str)
        burst: can be either a single burst metadata (dict) or a list of burst metadata (list[dict])
        force_type (str, optional): request a specific type of orbit file (can be 'orbres' or 'orbpoe')

    Returns
        str: the type of orbit found ('orbres' or 'orbpre')

    Raises
        FileNotFoundError: if no orbit file is found for the product_info
    '''

    def source(date, missionid, type):
        file = search_valid_orbit_files_from_local_folder(path, (date, missionid), type)
        if not file:
            return None

        return open(file, 'rb')

    return _update_statevectors_from_source(product_info, burst, force_type=force_type, source=source)


def update_statevectors_using_phoenix(phx_client, product_info, burst,
                                      *, force_type=None, phx_source="aws:proxima:kayrros-prod-sentinel-aux"):
    '''Retrieve the orbit statevectors of the given bursts using the Phoenix catalog.

    Args
        phx_client: phoenix client
        product_info: can be either a S1 SLC product_id (str) or a tuple containing the missionid (str) and the date (str)
        burst: can be either a single burst metadata (dict) or a list of burst metadata (list[dict])
        force_type (str, optional): request a specific type of orbit file (can be 'orbres' or 'orbpoe')
        phx_source (str, default to Kayrros Proxima): phoenix source from the esa-sentinel-1-csar-aux collection

    Returns
        str: the type of orbit found ('orbres' or 'orbpre')

    Raises
        FileNotFoundError: if no orbit file is found for the product_info
    '''

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

    return _update_statevectors_from_source(product_info, burst, force_type=force_type, source=source)
