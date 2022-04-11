import os
import numpy as np
from datetime import datetime

import sunpy.map as smp
from sunpy.coordinates import frames
import astropy.units as u
from astropy.coordinates import SkyCoord

class ts_processing:
    def __init__(self, path, ar_no):
        self.path = path
        self.ar_no = ar_no

    def process(self):
        # Read whole series
        TS_list_orig, ls_files_orig = self.get_magnetogram_files_twelve_min()

        # Check centroid
        TS_list_n, ls_files_n = self.check_SPEI(TS_list_orig, ls_files_orig)

        # Check corner
        TS_list_n, ls_files_n = self.patch_corner_check(TS_list_n, ls_files_n)

        return TS_list_n, ls_files_n

    def get_TS(self, filename):
        part = filename.split('.')[3]
        year = part[0:4]
        month = part[4:6]
        day = part[6:8]
        hr = part[9:11]
        minute = part[11:13]
        sec = part[13:15]
        iso = year + '-' + month + '-' + day + 'T' + hr + ':' + minute + ':' + sec
        TS = datetime.fromisoformat(iso)
        return TS

    def get_magnetogram_files_twelve_min(self):
        TS = {}
        TS_list = []
        magnetogram_list = []
        os.chdir(self.path + str(self.ar_no) + '/')
        files = os.listdir()

        magnetogram_files = sorted([file for file in files if 'magnetogram' in file])
        if magnetogram_files == None or magnetogram_files == []:
            return [], []

        for f in magnetogram_files:
            TS[f] = self.get_TS(f)
            magnetogram_list.append(smp.Map(f))

        return list(TS.values()), magnetogram_list

    # CRVAL1 = Carrington Longitude at the center of patch
    # CRVAL2 = Carrington Latitude at the center of patch
    # CRLN_OBS = Carrington Longitude of the Observer (at center of solar disk)
    # CRLT_OBS = Carrington Latitude of the Observer (at center of solar disk)
    def great_circle_distance(self, x, y):
        # Calculates the great circle distance (central angle) between
        # Point1: [lat, long]
        # Point2: [lat, long]  = [0, 0] if the angle is required wrt disk center
        # On the surface of a sphere with radius r
        # INPUT UNIT: DEGREES
        x = x * np.pi / 180
        y = y * np.pi / 180
        dlong = x[1] - y[1]
        den = np.sin(x[0]) * np.sin(y[0]) + np.cos(x[0]) * np.cos(y[0]) * np.cos(dlong)
        num = (np.cos(y[0]) * np.sin(dlong)) ** 2 + (
                    np.cos(x[0]) * np.sin(y[0]) - np.sin(x[0]) * np.cos(y[0]) * np.cos(dlong)) ** 2
        # Calculate the great circle distance:
        sig = np.arctan2(np.sqrt(num), den) * 180 / np.pi
        return sig

    def heliocentric_angle(self, CRVAL1, CRVAL2, CRLN_OBS, CRLT_OBS):
        # Calculate the Stonyhurst Latitude and Longitude:
        longitude = CRVAL1 - CRLN_OBS
        latitude = CRVAL2 - CRLT_OBS
        x = np.array([latitude, longitude])
        y = np.array([0., 0.])
        return self.great_circle_distance(x, y)

    # Function to calculate radial heliocentric angle of any coordinate on image:
    def get_hc_angle(self, header):
        crval1 = header['CRVAL1']
        crval2 = header['CRVAL2']
        crln_obs = header['CRLN_OBS']
        crlt_obs = header['CRLT_OBS']
        hc_angle = self.heliocentric_angle(crval1, crval2, crln_obs, crlt_obs)
        header['HC_ANGLE'] = hc_angle
        return hc_angle

    def check_SPEI(self, TS_list, m_files):
        if TS_list == [] or TS_list == None:
            return [], []

        list_of_files = []
        new_TS_list = []
        for time, value in zip(TS_list, m_files):
            angle = self.get_hc_angle(value.fits_header)
            if angle < 70:
                list_of_files += [value]
                new_TS_list += [time]

        return new_TS_list, list_of_files

    def patch_corner_check(self, TS_list, m_files):
        ls_map = []
        ls_n_TS = []

        if TS_list == [] or TS_list == None:
            return [], []

        else:
            for t, file in zip(TS_list, m_files):
                ob_time = file.fits_header['DATE-OBS']

                bl_coor = self.coord_transformer_hsc(file.bottom_left_coord.lon.deg, file.bottom_left_coord.lat.deg, ob_time)
                tr_coor = self.coord_transformer_hsc(file.top_right_coord.lon.deg, file.top_right_coord.lat.deg, ob_time)

                a_lat_lon = np.array([bl_coor.lat.deg, bl_coor.lon.deg])  # bottom_left
                b_lat_lon = np.array([tr_coor.lat.deg, tr_coor.lon.deg])  # top_right

                a_circle_dist = self.great_circle_distance(a_lat_lon, np.array([0., 0.]))  # bottom_left
                b_circle_dist = self.great_circle_distance(b_lat_lon, np.array([0., 0.]))  # top_right

                if a_circle_dist < 70 and b_circle_dist < 70:  # bottom left and top_right corner
                    ls_map.append(file)
                    ls_n_TS.append(t)

            return ls_n_TS, ls_map

    def coord_transformer_hgc(self, x, y, obs_time):
        x = x  # Stonyhurst longitude
        y = y  # Stonyhurst latitude

        c = SkyCoord(x * u.deg, y * u.deg, frame=frames.HeliographicStonyhurst, obstime=obs_time, observer="earth")  # )
        c_hgs = c.transform_to(frames.HeliographicCarrington)
        return c_hgs

    def coord_transformer_hsc(self, x, y, obs_time):
        x = x  # Stonyhurst longitude
        y = y  # Stonyhurst latitude

        c = SkyCoord(x * u.deg, y * u.deg, frame=frames.HeliographicCarrington, obstime=obs_time, observer="earth")  # )
        c_hgs = c.transform_to(frames.HeliographicStonyhurst)
        return c_hgs







