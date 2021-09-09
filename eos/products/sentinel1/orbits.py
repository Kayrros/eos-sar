import os
import io
import functools

from lxml import etree

from .metadata import string_to_timestamp

S1_ORBITS_BUCKET = 's1-orbits'


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


def select_orbit_file_from_filelist(files, date, missionid):
    date = string_to_timestamp(date)
    missionid = missionid.lower()

    for file in files:
        filename = os.path.basename(file)

        if filename[:len(missionid)].lower() != missionid:
            continue

        s, e = _parse_start_end_date_from_orbit_file(filename)
        s = string_to_timestamp(s)
        e = string_to_timestamp(e)

        # time buffer of 10 state vectors with 10 seconds per state vector before the date
        buffer_pre = 10 * 10
        # time buffer of 20 state vectors with 10 seconds per state vector after the date, since the date often indicates the beginning of the product
        buffer_post = 10 * 10

        if s + buffer_pre < date and e - buffer_post > date:
            return file

    raise FileNotFoundError(f'could not find an orbit file for date={date} mission={missionid}')


@functools.lru_cache
def _list_files_from_s3(client_s3, bucket, prefix):
    paginator = client_s3.get_paginator("list_objects_v2")

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj['Key']
            files.append(key[len(prefix):])

    return files


def select_orbit_file_from_s3(client_s3, bucket, prefix, date, missionid):
    files = _list_files_from_s3(client_s3, bucket, prefix)
    filenames = [os.path.basename(prefix + f) for f in files]
    filename = select_orbit_file_from_filelist(filenames, date, missionid)
    index = filenames.index(filename)
    return files[index]


def apply_new_statevectors_to_burst(xml_content, burst, orbtype):
    apply_new_statevectors_to_bursts(xml_content, [burst], orbtype)


def apply_new_statevectors_to_bursts(xml_content, bursts, orbtype):
    all_start = min([burst['state_vectors'][0]['time'] for burst in bursts])
    all_end = max([burst['state_vectors'][-1]['time'] for burst in bursts])

    newsvs = [[] for _ in bursts]

    if type(xml_content) == str:
        xml_content = io.BytesIO(xml_content.encode('utf-8'))

    context = etree.iterparse(xml_content, events=('end',), tag='OSV')
    for _, element in context:
        date = string_to_timestamp(element.findtext('UTC')[4:])

        if date < all_start - 10:
            continue
        if date > all_end + 10:
            break

        for i, b in enumerate(bursts):
            old_sv_start = b['state_vectors'][0]['time']
            old_sv_end = b['state_vectors'][-1]['time']

            if date < old_sv_start - 10:
                continue
            if date > old_sv_end + 10:
                continue

            newsvs[i].append({
                'time': date,
                'position': [float(element.findtext(k)) for k in ['X', 'Y', 'Z']],
                'velocity': [float(element.findtext(k)) for k in ['VX', 'VY', 'VZ']]
            })

    for i, b in enumerate(bursts):
        # make sure we fetched enough state_vectors
        assert len(b['state_vectors']) <= len(newsvs[i])
        b['state_vectors'] = newsvs[i]
        b['state_vectors_origin'] = orbtype


def update_statevectors_using_our_bucket(client_s3, product_info, burst, *, force_type=None):
    '''Retrieve the orbit statevectors of the given bursts using the bucket s3://s1-orbits.

    Args
        client_s3: a boto3.client instance with kayrros OIO credentials
        product_info: can be either a S1 SLC product_id (str) or a tuple containing the missionid (str) and the date (str)
        burst: can be either a single burst metadata (dict) or a list of burst metadata (list[dict])
        force_type (str, optional): request a specific type of orbit file (can be 'orbres' or 'orbpoe')

    Returns
        str: the type of orbit found ('orbres' or 'orbpre')

    Raises
        FileNotFoundError: if no orbit file is found for the product_info
    '''
    if isinstance(product_info, tuple):
        # ('20210216T151206', 'S1A')
        date, missionid = product_info
    else:
        # 'S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650'
        assert isinstance(product_info, str)
        missionid = product_info[:3]
        date = product_info[17:32]

    # compute relevant dates for a fast S3 listing
    curyear = int(date[:4])
    curmonth = int(date[4:6])
    nextyear = curyear + 1
    nextmonth = curmonth + 1 if curmonth < 12 else 1
    nextmonthyear = curyear if curmonth < 12 else nextyear
    curmonthdate = f'{curyear:04d}{curmonth:02d}'
    nextmonthdate = f'{nextmonthyear:04d}{nextmonth:02d}'

    def try_for_orbit_type(type):
        orbtype = f'orb{type}'
        prefix = f'{type}/{missionid.upper()}_OPER_AUX_{type.upper()}ORB_OPOD_'

        prefixes = (
            # first, look for the same month
            prefix + curmonthdate,
            # then, the next month
            prefix + nextmonthdate,
            # we could search more broadly, but that shouldn't be necessary?
        )

        for prefix in prefixes:
            try:
                file = select_orbit_file_from_s3(client_s3, S1_ORBITS_BUCKET, prefix, date, missionid)
                break
            except FileNotFoundError:
                pass
        else:
            return None

        xml = io.BytesIO()
        client_s3.download_fileobj(
            Fileobj=xml,
            Bucket=S1_ORBITS_BUCKET,
            Key=f'{prefix}{file}')
        xml.seek(0)

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
