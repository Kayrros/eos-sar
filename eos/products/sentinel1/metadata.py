"""Fill needed metadata of a burst."""
import os
import io
import math
import dateutil.parser
import datetime
import xmltodict
from lxml import etree
import numpy as np
from eos.sar import const


def string_to_timestamp(s):
    """Convert a string representing a date and time to a float number."""
    return dateutil.parser.parse(s).replace(
        tzinfo=datetime.timezone.utc).timestamp()


def corners_of_geolocation_grid_points_list(l, only_burst_id):
    """Return the 4 corners of a Sentinel-1 geolocation grid points list.\
    only_burst_id (int): restrict to a particular burst."""
    lines = sorted(list(set(int(c['line']) for c in l)))
    first_line_position = lines[only_burst_id]
    last_line_position = lines[only_burst_id+1]
    l = [c for c in l if int(c['line']) in (
        first_line_position, last_line_position)]
    line_indices = [int(c['line']) for c in l]
    first_line = [c for c in l if int(c['line']) == min(line_indices)]
    last_line = [c for c in l if int(c['line']) == max(line_indices)]
    a = min(first_line, key=lambda k: int(k['pixel']))
    b = max(first_line, key=lambda k: int(k['pixel']))
    c = max(last_line, key=lambda k: int(k['pixel']))
    d = min(last_line, key=lambda k: int(k['pixel']))
    return a, b, c, d


def _compute_burst_id(o, i, b, first_burst):
    """Compute the absolute/relative burst id and adds theses entries to the
       provided dictionary.
    The absolute burst id (+ subswath) provides a unique identifier for a
    Sentinel 1 burst, while the relative burst id (+ subswath) provides an
    identifier that is the same for each burst looking at similar footprints.

    Parameters
    ----------
    o: dict
       Metadata computed for the current burst.
    i: dict
       Data from the xml annotation file.
    b: dict
       Data of the current burst directly extracted from the xml annotation file.

    Raises
    ------
    ValueError
        In case mission_id in the provided dict is not S1A/S1B.
    ValueError
        In case the swath in the provided dict is not IW1/IW2/IW3.

    Returns
    -------
    None.
    """
    # compute the orbit numbers
    absolute_orbit_number = int(i['adsHeader']['absoluteOrbitNumber'])
    mission_id = i['adsHeader']['missionId']
    if mission_id == 'S1A':
        relative_orbit_number = (absolute_orbit_number - 73) % 175 + 1
    elif mission_id == 'S1B':
        relative_orbit_number = (absolute_orbit_number - 27) % 175 + 1
    else:
        raise ValueError(f'Invalid mission_id {mission_id}')

    # time taken to go over one orbit
    # repeat cycle is 12 days, with 175 orbits per cycle
    T_orb = 12 * 24 * 3600 / 175

    # T_beam is sensing time between two bursts
    # this value was obtained by subtracting the sensing time of two consecutive bursts
    # and tweaking to fit the ground-truth burst id data
    T_beam = 2.75827302

    # T_pre is a time offset to account for the fact that the
    # first burst of each swaths crosses the ascending node
    # at different times (and might not at t=0)
    if o['swath'] == 'IW1':
        T_pre = 0.8
    elif o['swath'] == 'IW2':
        T_pre = 1
    elif o['swath'] == 'IW3':
        T_pre = 2
    else:
        raise ValueError(f"Invalid subswath {o['swath']}")

    # compute the mid-burst sensing time
    N = o['lines_per_burst']
    PRF = o['azimuth_frequency']
    # difference between the mid-burst time and the time at which the orbit crossed the ascending node
    t_anx = float(b['azimuthAnxTime'])
    # time at the middle of the burst
    delta_b = t_anx + (N - 1) / (2 * PRF)
    # subtract a short delay to align the swaths
    delta_b -= T_pre

    # in the following, we have 3 slices (= 3 products), 3 bursts per slice
    # the second slice crosses the equator, somewhere inside the second burst
    # ================
    # |p3 b3         |
    # |              |
    # |              |
    # |--------------|
    # |p3 b2         |
    # |              |
    # |              |
    # |--------------|
    # |p3 b1         |
    # |              |
    # |              |
    # ================
    # |p2 b3         |
    # |              |
    # |              |
    # |--------------|
    # |p2 b2         | orbit=2 above
    # //////////////// equator          <- t=tANX+T_orb
    # |              | orbit=1 below
    # |--------------|
    # |p2 b1         |
    # |              |
    # |              |
    # ================
    # |p1 b3         |
    # |              |
    # |              |
    # |--------------|
    # |p1 b2         |
    # |              |
    # |              |
    # |--------------|
    # |p1 b1         |
    # |              |
    # |              |
    # ================
    # ...
    #                  orbit=1 above
    # //////////////// equator          <- t=tANX
    #                  orbit=0 below

    # looking at the xmls, we have the following:
    # - the orbit number is updated only at the beginning of the product
    #       orbit number for product 0 is 1
    #       orbit number for product 1 is 1
    #       orbit number for product 2 is 2
    # - ascendingNodeTime is only updated at the beginning of the first slice
    #       all products have ascendingNodeTime=tANX
    #       in the burst metadata, azimuthAnxTime is computed from ascendingNodeTime
    #           this means that for p2-b2 and beyond, their time-since-anx will be greater than the orbit duration
    #           even though the orbit number of bursts of p3 is already incremented
    # this means that we have to take extra care and retrieve the correct orbit number or ANX time for each case

    # the longitude at the start of the first orbit is 4.5° at the equator
    orbit1_lon = 4.5
    # the angle difference between two consecutive orbits
    angle_per_orbit = 360 / 175 * 12
    # approximative longitude of the current burst (it's ok if it is a few degrees of)
    current_lon = np.mean([c[0] for c in o['approx_geom']])
    # expected longitude for a given orbit number
    lon_at_anx = lambda orbit: (orbit1_lon - angle_per_orbit * (orbit - 1) + 180) % 360 - 180
    # since the longitude difference between the swaths is +/-1° the longitude of swath 2,
    # we don't have to correct it since the margin of error will be around 10°
    expected_lon = lon_at_anx(relative_orbit_number)

    # the current orbit is wrong if
    # 1. we are close to the next ANX
    # 2. the orbit number was not yet incremented (= expected_lon is way off)
    close_to_next_anx = abs(t_anx - T_orb) < 50
    orbit_looks_off = abs(current_lon - expected_lon) > angle_per_orbit / 2

    # if the ANX of the first burst is larger than the orbit time, then we crossed the equator
    # two cases can happen:
    # 1. the orbit number was already updated, then we need to reset ANX
    #    this happens when the product is fully after the equator
    # 2. the orbit number is the old one, then we don't have to do anything
    #    this happens when the product is crossing the equator
    # - to avoid mis-reseting the ANX, we detect case 2. by checking whether
    #   we are close to the ANX and if the longitude is coherent with the current orbit
    #   the old orbit will predict a longitude off by ~25° (= angle_per_orbit)
    # - then we can check case 1. by checking whether the first burst of the product
    #   was already after the next anx
    # this strategy avoid having to rely on precise timings (with timings from other swaths to consider)
    fanx = float(first_burst['azimuthAnxTime'])
    if not (close_to_next_anx and orbit_looks_off) and fanx > T_orb:
        delta_b -= T_orb

    # NOTE: if we completed an orbit during the acquisition, we would need to adjust the orbit numbers
    #       this is only required because of the relative_orbit_number
    #       however, the start of orbit 1 is above the sea so we can ignore this case for now

    # add the time taken to arrive to the current orbit
    delta_t_b_rel = delta_b + (relative_orbit_number - 1) * T_orb
    delta_t_b_abs = delta_b + (absolute_orbit_number - 1) * T_orb

    # subtract the preamble and divide by the beam cycle time to obtain the burst ids
    o['relative_burst_id'] = 1 + math.floor(delta_t_b_rel / T_beam)
    o['absolute_burst_id'] = 1 + math.floor(delta_t_b_abs / T_beam)

def extract_common_metadata(xml):
    """Extract common metadata for all the swath.

    Parameters
    ----------
    xml : str
        Content of the xml annotation file.

    Returns
    -------
    o : dict
        Extracted metadata from the xml.
    i : dict
        Full metadata contained in the xml as a nested dictionnary.

    """
    i = xmltodict.parse(xml)['product']  # input full dictionary (huge)
    o = {}  # output dictionary with only the stuff we need (tiny)

    d = i['imageAnnotation']['imageInformation']
    o['azimuth_frequency'] = float(d['azimuthFrequency'])
    o['slant_range_time'] = float(d['slantRangeTime'])

    d = i['swathTiming']
    o['lines_per_burst'] = int(d['linesPerBurst'])
    o['samples_per_burst'] = int(d['samplesPerBurst'])

    # subswath of the burst
    o['swath'] = i['adsHeader']['swath']

    d = i['generalAnnotation']['productInformation']
    o['range_frequency'] = float(d['rangeSamplingRate'])
    o['orbit_pass'] = d['pass']

    # state vectors (sv)
    o['state_vectors'] = []
    for s in i['generalAnnotation']['orbitList']['orbit']:
        o['state_vectors'].append({
            'time': string_to_timestamp(s['time']),
            'position': [float(s['position'][k]) for k in ['x', 'y', 'z']],
            'velocity': [float(s['velocity'][k]) for k in ['x', 'y', 'z']]
        })
    # we assume the lowest quality here, even though some products might be generated with more accurate orbit data
    o['state_vectors_origin'] = 'orbpre'

    # deramping parameters
    o['steering_rate'] = np.radians(
        float(i['generalAnnotation']['productInformation']['azimuthSteeringRate']))
    o['wave_length'] = const.LIGHT_SPEED_M_PER_SEC / \
        float(i['generalAnnotation']['productInformation']['radarFrequency'])

    # azimuth fm rates
    o['az_fm_times'] = []
    o['az_fm_info'] = []
    for az in i['generalAnnotation']['azimuthFmRateList']['azimuthFmRate']:
        try:
            azp = az['azimuthFmRatePolynomial']['#text'].split()
        except KeyError:  # old xml files were formatted differently
            azp = [az['c0'], az['c1'], az['c2']]
        o['az_fm_times'].append(string_to_timestamp(az['azimuthTime']))
        o['az_fm_info'].append(
            list(map(float, [az['t0'], azp[0], azp[1], azp[2]])))

    # doppler centroid estimates
    dc_estimate = i['dopplerCentroid']['dcEstimateList']['dcEstimate']
    o['dc_estimate_time'] = [string_to_timestamp(
        x['azimuthTime']) for x in dc_estimate]
    o['dc_estimate_t0'] = [float(x['t0']) for x in dc_estimate]
    if i['imageAnnotation']['processingInformation']['dcMethod'] == 'Data Analysis':
        dc_polynomial_name = 'dataDcPolynomial'
    else:  # geometrical method. Polynom more stable
        dc_polynomial_name = 'geometryDcPolynomial'
    o['dc_estimate_poly'] = []
    for x in dc_estimate:
        o['dc_estimate_poly'].append(
            list(map(float, x[dc_polynomial_name]['#text'].split())))
    return o, i


def extract_bursts_metadata(xml, burst_ids=None):
    """Extract metadata for a list of bursts.

    Parameters
    ----------
    xml : str
        Content of the xml annotation file..
    burst_ids : Iterable, optional
        Ids of the bursts to process. If None (not given), the metadata of all
        the bursts in the swath will be returned. The default is None.

    Returns
    -------
    bursts: List of dicts
        The metadata of the bursts.
    """
    o, i = extract_common_metadata(xml)

    dictbursts = i['swathTiming']['burstList']['burst']

    if burst_ids:
        assert min(burst_ids) >= 0 and max(burst_ids) < len(dictbursts),\
            "burst ids out of range"
    else:
        burst_ids = range(len(dictbursts))

    # longitude, latitude bounding box: select the four corners of the gcp grid
    gcp = i['geolocationGrid']['geolocationGridPointList']['geolocationGridPoint']

    bursts = []
    for bid in burst_ids:
        b = dictbursts[bid]

        # the metadata of the burst contains also the common metadata of the swath
        burst = o.copy()

        # region of interest (roi) within the burst: x, y, w, h
        first_valid_x = map(int, b['firstValidSample']['#text'].split())
        last_valid_x = map(int, b['lastValidSample']['#text'].split())
        valid_rows_left = [(v, j)
                           for j, v in enumerate(first_valid_x) if v >= 0]
        valid_rows_right = [(v, j)
                            for j, v in enumerate(last_valid_x) if v >= 0]
        x, y = valid_rows_left[0]
        w = valid_rows_right[0][0] - x + 1
        h = valid_rows_left[-1][1] - y + 1

        # time interval corresponding to the valid burst domain
        start = string_to_timestamp(b['azimuthTime'])
        start_valid = start + y / o['azimuth_frequency']
        end_valid = start_valid + h / o['azimuth_frequency']
        burst['burst_times'] = (start, start_valid, end_valid)

        # make the burst roi coordinates relative the the full tiff image
        y += bid * o['lines_per_burst']
        burst['burst_roi'] = (x, y, w, h)

        burst['azimuth_anx_time'] = float(b['azimuthAnxTime'])

        corners = corners_of_geolocation_grid_points_list(
            gcp, only_burst_id=bid)
        burst['approx_geom'] = [(float(c['longitude']),
                                       float(c['latitude'])) for c in corners]

        # compute the burst id
        _compute_burst_id(burst, i, b, dictbursts[0])

        bursts.append(burst)

    return bursts

def extract_burst_metadata(xml, burst_id):
    """Extract metadata for a single burst.

    Parameters
    ----------
    xml : str
        Content of the xml annotation file.
    burst_id : Integer
        Id of the burst to process.

    Returns
    -------
    b_dicts: dicts
        The metadata of the burst.
    """
    return extract_bursts_metadata(xml, burst_ids=[burst_id])[0]

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
        b['state_vectors'] = newsvs[i]
        b['state_vectors_origin'] = orbtype

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
    missionid = missionid.lower()

    for file in files:
        filename = os.path.basename(file)

        if filename[:len(missionid)].lower() != missionid:
            continue

        s, e = _parse_start_end_date_from_orbit_file(filename)
        if s < date and e > date:
            return file

    raise FileNotFoundError(f'could not find an orbit file for {date}/{missionid}')

