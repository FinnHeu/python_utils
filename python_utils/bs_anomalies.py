import numpy as np
import xarray as xr
from python_utils.dataset_operations import *

def pressure_to_geowind(pressure, lon, lat, rho = 1.225, Re = 6371000, f='f-plane', f_central=None, slp=False):

    '''
    pressure_to_geowind.py

    Computes geostrophic winds from sea level pressure field

    Inputs
    ------
    pressure: np.ndarray, sea level pressure field in Pascal
    lon: np.ndarray, longitude vector
    lat: np.ndarray, latitude vector
    rho = 1.225, air density
    Re = 6371000, radius earth
    f='f-plane' or 'real_world'
    f_central=None, central latitude for f-plane
    slp = False, correct for surface friction (Protushinsky and Johnson 1997) when using sea level pressure fields

    Returns
    -------
    u, v: zonal and meridional wind components
    '''

    # Compute the distances between the gridpoints for computing the gradient
    dist_x = np.diff(lon)[0] * (2 * np.pi * Re / 360) * np.cos(lat/360 * 2 * np.pi)
    dist_y = np.diff(lat) * (2 * np.pi * Re / 360)

    # Compute the gradient
    dpdx = np.ones_like(pressure) * np.nan
    dpdy = np.ones_like(pressure) * np.nan

    for i in range(len(lat)): dpdx[i,:] = np.gradient(pressure[i,:], dist_x[i])
    for i in range(len(lon)): dpdy[:,i] = np.gradient(pressure[:,i], np.mean(dist_y))

    # Compute Coriolis parameter
    if f == 'f-plane':
        if f_central == None: f_c = 2 * 7.2921e-5 * np.sin(np.median(lat))
        else: f_c = 2 * 7.2921e-5 * np.sin(np.median(f_central))

        f_c = np.ones_like(dpdx) * f_c

    elif f == 'real_world':
        f_c = 2 * 7.2921e-5 * np.sin(lat)
        f_c = f_c.repeat(len(lon)).reshape(dpdx.shape)

    # Compute geostrophic velocity
    u = 1/(rho * f_c) * dpdy
    v = -1/(rho * f_c) * dpdx

    if slp:
        # Deflect and reduce winds for the impact of surface friction (Protushinsky and Johnson 1997)
        rad = np.pi/180
        scale = .7
        angle = 30

        rot_mat = np.array([[np.cos(angle*rad), -np.sin(angle*rad)],[np.sin(angle*rad), np.cos(angle*rad)]])

        a = u.shape
        for i in range(a[0]):
            for j in range(a[1]):

                vec = np.array([u[i,j], v[i,j]]) # extract velocity vector
                rot_vec = scale * np.dot(rot_mat, vec) #  rotate and scale

                u[i,j] = rot_vec[0]
                v[i,j] = rot_vec[1]


    return u, v


def mean_diff_two_periods(src, period1=('1979','1999'), period2=('2000','2018'), param='slp', winter_only=False, lon180=False, scale=1):

    '''
    anomaly_two_periods.py

    Computes the difference between two reference periods in one dataset.
    Computation: period2 - period1

    Inputs
    ------
    src (str or xr.DataArray or xr.Dataset)
        path to dataset or dataset as xr.DataArray / xr.Dataset
    period1, period2 (tuple)
        tuples with start and end of each period
    winter_only (bool, list)
        if True only DJFMAM are considered, if list month in list are considered
    scale (int, float)
        scale factor (default=1)


    Returns
    -------
    ds_p1

    ds_p2

    delta_ds
        dataset containing the subtracted periods

    '''

    # Handle different input cases for src_path
    if isinstance(src, str):
        # Open file
        ds = xr.open_dataset(src).load()

    elif isinstance(src, xr.DataArray):
        ds = src.to_dataset().load()

    elif isinstance(src, xr.Dataset):
        ds = src.load()

    # Apply cf conventions
    ds = dataset_to_cfconvention(ds, lon180=lon180)

    # Select month
    if winter_only:
        if isinstance(winter_only, bool):
            ds = select_winter_month(ds)
        elif isinstance(winter_only, list):
            ds = select_winter_month(ds, month=winter_only)


    # Select two time periods
    ds_p1 = ds.sel(time=slice(period1[0], period1[1]))
    ds_p2 = ds.sel(time=slice(period2[0], period2[1]))

    print('First period: ' + str(ds_p1.time[0].values), ' to ' + str(ds_p1.time[-1].values))
    print('First period: ' + str(ds_p2.time[0].values), ' to ' + str(ds_p2.time[-1].values))

    # time mean
    ds_p1 = ds_p1.mean(dim='time')
    ds_p2 = ds_p2.mean(dim='time')

    # Compute the difference of the mean fields and scale
    ds_delta = (ds_p2[param] - ds_p1[param]) * scale

    return ds_p1, ds_p2, ds_delta


def create_wind_anomaly_netCDF(ds, lon_range, lat_range, lon360=True, savepath=None):
    '''
    Create_wind_anomaly_netCDF.py

    Creates global netCDF files with wind anomalies for use in fesom2.1

    Inputs
    ------
    ds, xr.dataarray
        slp data
    lon_range, tuple
        (min_lon, max_lon)
    lat_range, tuple
        (min_lat, max_lat)
    lon360, bool
        convert longitudes to 0-360° range (default is True)
    savepath, string
        path to save the netCDF file

    Returns
    -------
     ds_new, xr.dataset
         global dataset with wind anomalies

    '''

    # Find the nearest values to the given ranges
    lon_min = ds.lon.sel(lon=lon_range[0], method='nearest').values
    lon_max = ds.lon.sel(lon=lon_range[-1], method='nearest').values
    lat_min = ds.lat.sel(lat=lat_range[0], method='nearest').values
    lat_max = ds.lat.sel(lat=lat_range[-1], method='nearest').values

    # Compute geostrophic wind
    slp = ds.values
    lon = ds.lon.values
    lat = ds.lat.values

    u, v = pressure_to_geowind(slp, lon, lat, f='f-plane', f_central=75, slp=True)


    # Set all values outside range to 0
    LON, LAT = np.meshgrid(lon,lat)

    u_anom = np.where((LON >= lon_min) & (LON <= lon_max) & (LAT >= lat_min) & (LAT <= lat_max), u, 0)
    v_anom = np.where((LON >= lon_min) & (LON <= lon_max) & (LAT >= lat_min) & (LAT <= lat_max), v, 0)

    ds_new = xr.Dataset({
    'uanom': xr.DataArray(
                data   = u_anom,   # enter data here
                dims   = ['lat', 'lon'],
                coords = {'lat': lat, 'lon': lon},
                attrs  = {
                    'units'     : 'm/s',
                    'description': 'gesostrophic wind anomaly from JRA55 slp difference 1979-1999, 2000-2018, Rotated by 30° and scaled by a factor of 0.7 (Protoshinsky and Johnson 1997)'
                    }
                ),
     'vanom': xr.DataArray(
                data   = v_anom,   # enter data here
                dims   = ['lat', 'lon'],
                coords = {'lat': lat, 'lon': lon},
                attrs  = {
                    'units'     : 'm/s',
                    'description': 'gesostrophic wind anomaly from JRA55 slp difference 1979-1999, 2000-2018, Rotated by 30° and scaled by a factor of 0.7 (Protoshinsky and Johnson 1997)'
                    }
                ),
    })

    if lon360:
        ds_new.coords['lon'] = np.where(ds_new.lon.values < 0, ds_new.lon.values + 360, ds_new.lon.values)
        ds_new = ds_new.sortby(ds_new.lon)

    # Add Time coordinate
    ds_new = ds_new.assign_coords({'time': 1}).expand_dims('time')

    if savepath:
        ds_new.to_netcdf(savepath)

    return ds_new
