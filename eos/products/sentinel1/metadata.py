"""Fill needed metadata of a burst."""
import math
import dateutil.parser
import datetime
import xmltodict
import numpy as np
import logging

from eos.sar import const

logger = logging.Logger(__name__)

# time taken to go over one orbit
# repeat cycle is 12 days, with 175 orbits per cycle
N_orbits_per_cycle = 175
T_orb = 12 * 24 * 3600 / N_orbits_per_cycle

# These two constants were provided by (ESA)
T_beam = 2.758273
T_pre = 2.299849

# compute a T_orb2 for which the number of bursts is an integer
N_bursts_per_cycle = 375887
T_orb2 = T_beam * N_bursts_per_cycle / N_orbits_per_cycle


def string_to_timestamp(s):
    """Convert a string representing a date and time to a float number."""
    return dateutil.parser.parse(s).replace(
        tzinfo=datetime.timezone.utc).timestamp()


def corners_of_geolocation_grid_points_list(l, only_burst_id):
    """Return the 4 corners of a Sentinel-1 geolocation grid points list.\
    only_burst_id (int): restrict to a particular burst."""
    lines = sorted(list(set(int(c['line']) for c in l)))
    first_line_position = lines[only_burst_id]
    last_line_position = lines[only_burst_id + 1]
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


def _mid_burst_sensing_time_correction(o, first_burst_xml):
    """
    """
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

    def lon_at_anx(orbit): return (orbit1_lon - angle_per_orbit * (orbit - 1) + 180) % 360 - 180
    # since the longitude difference between the swaths is +/-1° the longitude of swath 2,
    # we don't have to correct it since the margin of error will be around 10°
    expected_lon = lon_at_anx(o['relative_orbit_number'])

    # the current orbit is wrong if
    # 1. we are close to the next ANX
    # 2. the orbit number was not yet incremented (= expected_lon is way off)
    t_anx = o['azimuth_anx_time']
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
    fanx = float(first_burst_xml['azimuthAnxTime'])
    if not (close_to_next_anx and orbit_looks_off) and fanx > T_orb:
        return -T_orb
    else:
        return 0

    # NOTE: if we completed an orbit during the acquisition, we would need to adjust the orbit numbers
    #       this is only required because of the relative_orbit_number
    #       however, the start of orbit 1 is above the sea so we can ignore this case for now


def compute_burst_id(burst, first_burst_xml):
    """Compute relative and absolute burst IDs.

    The absolute burst id (+ subswath) provides a unique identifier for a
    Sentinel-1 burst, while the relative burst id (+ subswath) provides an
    identifier that is the same for each burst looking at similar footprints.

    This function implements the formulas described in section 9.25 of the
    Sentinel-1 Level 1 Detailed Algorithm Definition (page 9-39).

    https://sentinels.copernicus.eu/documents/247904/4629273/Sentinel-1-Level-1-Detailed-Algorithm-Definition-v2-3.pdf/3ca88b95-770a-37c3-1b45-0031869d344a

    Parameters
    ----------
    burst: dict
       Metadata of the burst.

    Returns
    -------
    int: relative burst id
    int: absolute burst id
    """
    # orbit numbers, ANX time and burst parameters
    relative_orbit_number = burst['relative_orbit_number']
    absolute_orbit_number = burst['absolute_orbit_number']
    anx_time = burst['anx_time']
    n = burst['lines_per_burst']
    pri = burst['pri']
    burst_sensing_time = burst['burst_sensing_time'] + _mid_burst_sensing_time_correction(burst, first_burst_xml)

    # mid-burst sensing time
    t_b = burst_sensing_time + pri * (n - 1) / 2

    # time distance between t_b and the first ANX time in the current mission cycle
    delta_t_b_rel = t_b - anx_time + (relative_orbit_number - 1) * T_orb
    delta_t_b_abs = delta_t_b_rel + (absolute_orbit_number - relative_orbit_number) * T_orb2

    # subtract the preamble and divide by the beam cycle time to obtain the burst ids
    relative_burst_id = 1 + math.floor((delta_t_b_rel - T_pre) / T_beam) % N_bursts_per_cycle
    absolute_burst_id = 1 + math.floor((delta_t_b_abs - T_pre) / T_beam)

    return relative_burst_id, absolute_burst_id


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

    # compute orbit numbers
    mission_id = i['adsHeader']['missionId']
    absolute_orbit_number = int(i['adsHeader']['absoluteOrbitNumber'])

    if mission_id == 'S1A':
        relative_orbit_number = (absolute_orbit_number - 73) % 175 + 1
    elif mission_id == 'S1B':
        relative_orbit_number = (absolute_orbit_number - 27) % 175 + 1
    else:
        raise ValueError(f'Invalid mission_id {mission_id}')

    o['mission_id'] = mission_id
    o['absolute_orbit_number'] = absolute_orbit_number
    o['relative_orbit_number'] = relative_orbit_number

    d = i['imageAnnotation']['imageInformation']
    o['azimuth_frequency'] = float(d['azimuthFrequency'])
    o['slant_range_time'] = float(d['slantRangeTime'])
    o['anx_time'] = string_to_timestamp(d['ascendingNodeTime'])
    o['slice_number'] = int(d['sliceNumber'])

    d = i['swathTiming']
    o['lines_per_burst'] = int(d['linesPerBurst'])
    o['samples_per_burst'] = int(d['samplesPerBurst'])

    # subswath
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

    if i['adsHeader']['productType'] == 'SLC':
        # pulse things
        d = i['generalAnnotation']['downlinkInformationList']['downlinkInformation']['downlinkValues']
        o['chirp_rate'] = float(d['txPulseRampRate'])  # used for intra_pulse_correction
        o['pri'] = float(d['pri'])    # used for full_bistatic_correction
        o['rank'] = float(d['rank'])  # used for full_bistatic_correction

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

        # make the burst roi coordinates relative the the full swath
        y += bid * o['lines_per_burst']
        burst['burst_roi'] = (x, y, w, h)

        burst['azimuth_anx_time'] = float(b['azimuthAnxTime'])
        burst['burst_sensing_time'] = string_to_timestamp(b['sensingTime'])

        corners = corners_of_geolocation_grid_points_list(gcp,
                                                          only_burst_id=bid)
        burst['approx_geom'] = [(float(c['longitude']),
                                 float(c['latitude'])) for c in corners]
        burst['approx_altitude'] = [float(c['height']) for c in corners]

        # compute the burst id
        relative_burst_id, absolute_burst_id = compute_burst_id(burst, first_burst_xml=dictbursts[0])

        if 'burstId' in b:  # True for IPF >= 3.40
            esa_relative_burst_id = int(b['burstId']['#text'])
            if relative_burst_id != esa_relative_burst_id:
                logger.warning('relative_burst_id mismatch (xml:{}, computed:{})'
                               .format(esa_relative_burst_id, relative_burst_id))
            # don't compare the absolute burst id since our definition is different than ESA's one

        burst['relative_burst_id'] = relative_burst_id
        burst['absolute_burst_id'] = absolute_burst_id
        swath = burst['swath']
        burst['bsid'] = f'{relative_burst_id}_{swath}'

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


def extract_grd_metadata(xml):
    """Extract metadata for a GRD product.

    Parameters
    ----------
    xml : str
        Content of the xml annotation file.

    Returns
    -------
    b_dicts: dicts
        The metadata of the product.
    """
    o, i = extract_common_metadata(xml)

    gcp = i['geolocationGrid']['geolocationGridPointList']['geolocationGridPoint']
    corners = corners_of_geolocation_grid_points_list(gcp, 0)
    o['approx_geom'] = [(float(c['longitude']), float(c['latitude'])) for c in corners]
    o['approx_altitude'] = [float(c['height']) for c in corners]

    d = i['imageAnnotation']['imageInformation']
    o['image_start'] = string_to_timestamp(d['productFirstLineUtcTime'])
    o['image_end'] = string_to_timestamp(d['productLastLineUtcTime'])

    o['azimuth_time_interval'] = float(d['azimuthTimeInterval'])
    o['range_pixel_spacing'] = float(d['rangePixelSpacing'])

    srgr = i['coordinateConversion']['coordinateConversionList']['coordinateConversion']
    o['srgr'] = {
        'times': [string_to_timestamp(s["azimuthTime"]) for s in srgr],
        'srgr_coeffs': [list(map(float, srgr[k]["srgrCoefficients"]["#text"].split())) for k in range(len(srgr))],
        'grsr_coeffs': [list(map(float, srgr[k]["grsrCoefficients"]["#text"].split())) for k in range(len(srgr))],
        'sr0': [float(srgr[k]["sr0"]) for k in range(len(srgr))],
        'gr0': [float(srgr[k]["gr0"]) for k in range(len(srgr))],
    }

    o['width'] = int(i['imageAnnotation']['imageInformation']['numberOfSamples'])
    o['height'] = int(i['imageAnnotation']['imageInformation']['numberOfLines'])

    return o


def assemble_multiple_products_into_metas(metas_per_product):
    bursts = list(sum(metas_per_product, []))
    return bursts


def _unique_sv(state_vectors: list[dict]):
    """
    Get a list of unique state vectors from a list of redundant state_vectors.

    Parameters
    ----------
    state_vectors : list[dict]
        Each state vector is a dict with time, position, velocity of satellite.
        Here, state vectors may be duplicates.

    Returns
    -------
    unique_state_vectors : list[dict]
        Each state vector is a dict with time, position, velocity of satellite.
        Here state vectors have been filtered and are unique.

    """
    state_vectors = sorted(state_vectors, key=lambda x: x["time"])

    unique_state_vectors = [state_vectors[0]]
    for sv in state_vectors[1:]:
        if sv["time"] - unique_state_vectors[-1]["time"]:
            # different sample
            unique_state_vectors.append(sv)

    return unique_state_vectors


def unique_sv_from_bursts_meta(bursts_meta: list[dict]):
    """
    Get an aggregated list of state_vectors from bursts_meta

    Parameters
    ----------
    bursts_meta : list[dict]
        List of bursts metadata.

    Returns
    -------
    unique_state_vectors: list[dict]
        Each element is a unique state_vectors
    """
    state_vectors = [sv for bmeta in bursts_meta for sv in bmeta["state_vectors"]]
    return _unique_sv(state_vectors)
