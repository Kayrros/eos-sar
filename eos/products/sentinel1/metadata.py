"""Fill needed metadata of a burst."""
import math
import dateutil.parser
import datetime
import xmltodict
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


def _compute_burst_id(o, i, b):
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

    # time at which the ascending node was crossed for the current orbit
    orbit_anx_time = i['imageAnnotation']['imageInformation']['ascendingNodeTime']
    orbit_anx_time = datetime.datetime.fromisoformat(orbit_anx_time)

    # time taken to go over one orbit
    # repeat cycle is 12 days, with 175 orbits per cycle
    T_orb = 12*24*3600/175

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
    t_start = datetime.datetime.fromisoformat(b['azimuthTime'])
    N = o['lines_per_burst']
    PRF = o['azimuth_frequency']
    t_b = t_start + datetime.timedelta(seconds=(N - 1) / (2 * PRF))

    # compute the difference between the mid-burst time and
    # the time at which the orbit crossed the ascending node
    delta_b = (t_b - orbit_anx_time).total_seconds()

    # add the time taken to arrive to the current orbit
    delta_t_b_rel = delta_b + (relative_orbit_number - 1) * T_orb
    delta_t_b_abs = delta_b + (absolute_orbit_number - 1) * T_orb

    # subtract the preamble and divide by the beam cycle time
    # to obtain the burst ids
    o['relative_burst_id'] = 1 + math.floor((delta_t_b_rel - T_pre) / T_beam)
    o['absolute_burst_id'] = 1 + math.floor((delta_t_b_abs - T_pre) / T_beam)


def fill_meta(xml, bid):
    """
    Return a dictionary containing the data of some Sentinel-1 xml fields.

    Parameters
    ----------
        xml (string): content of a whole xml file

    Returns
    -------
        dictionary containing some of the xml data
    """
    i = xmltodict.parse(xml)['product']  # input full dictionary (huge)
    o = {}  # output dictionary with only the stuff we need (tiny)
    d = i['imageAnnotation']['imageInformation']
    o['azimuth_frequency'] = float(d['azimuthFrequency'])
    o['slant_range_time'] = float(d['slantRangeTime'])

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

    # longitude, latitude bounding box: select the four corners of the gcp grid
    gcp = i['geolocationGrid']['geolocationGridPointList']['geolocationGridPoint']
    d = i['swathTiming']
    o['lines_per_burst'] = int(d['linesPerBurst'])
    o['samples_per_burst'] = int(d['samplesPerBurst'])
    bursts = d['burstList']['burst']
    b = bursts[bid]
    # region of interest (roi) within the burst: x, y, w, h
    first_valid_x = map(int, b['firstValidSample']['#text'].split())
    last_valid_x = map(int, b['lastValidSample']['#text'].split())
    valid_rows_left = [(v, i)
                       for i, v in enumerate(first_valid_x) if v >= 0]
    valid_rows_right = [(v, i)
                        for i, v in enumerate(last_valid_x) if v >= 0]
    x, y = valid_rows_left[0]
    w = valid_rows_right[0][0] - x + 1
    h = valid_rows_left[-1][1] - y + 1

    # time interval corresponding to the valid burst domain
    start = string_to_timestamp(b['azimuthTime'])
    start_valid = start + y / o['azimuth_frequency']
    end_valid = start_valid + h / o['azimuth_frequency']
    o['burst_times'] = (start, start_valid, end_valid)

    # make the burst roi coordinates relative the the full tiff image
    y += bid * o['lines_per_burst']
    o['burst_roi'] = (x, y, w, h)

    o['azimuth_anx_time'] = float(b['azimuthAnxTime'])

    corners = corners_of_geolocation_grid_points_list(
        gcp, only_burst_id=bid)
    o['approx_geom'] = [(float(c['longitude']),
                         float(c['latitude'])) for c in corners]

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

    # subswath of the burst
    swath = i['adsHeader']['swath']
    o['swath'] = swath

    _compute_burst_id(o, i, b)

    return o

