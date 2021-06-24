import numpy as np
import xarray as xr

def pressure_to_geowind(pressure, lon, lat, rho = 1.225, Re = 6371000, f='f-plane', f_central=None, slp=False):

    '''
    pressure_to_geowind.py

    Computes geostrophic winds from sea level pressure field

    Inputs
    ------
    pressure: np.ndarray, sea level pressure field
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


def dataset_to_cfconvention(ds, longitude='lon', latitude='lat', time='time', slp='slp', temp='temp', salt='salt', lon180=True):

    '''
    dataset_to_cfconvention.py

    Renames the coordinates of an xarray dataset to lon, lat, time, slp, temp (according to cf conventions)

    Inputs
    ------
    ds (xr.dataset, xr.datarray)
        dataset to process
    varnames (str)
        new names of various coordinates and variables
    lon180 (bool)
        if True the logitude will be changes to -180E to 180E



    Returns
    -------
    ds (xr.dataset, xr.datarray)
        dataset with renamed coordinates/ variables

    '''

    # Rename Coordinates

    time_names = ['TIME','Time']
    lon_names = ['LONGITUDE', 'Longitude', 'LON', 'Lon', 'LONS', 'lons']
    lat_names = ['LATITUDE', 'Latitude', 'LAT', 'Lat', 'LATS', 'lats']

    coords = list(ds.coords)
    for coord_name in coords:

        for time_name in time_names:
            if (coord_name == time_name):
                ds = ds.rename({coord_name: time})

        for lon_name in lon_names:
            if (coord_name == lon_name):
                ds = ds.rename({coord_name: longitude})

        for lat_name in lat_names:
            if (coord_name == lat_name):
                ds = ds.rename({coord_name: latitude})


    # Rename Data variables

    slp_names = ['SLP', 'Slp', 'PSL', 'psl', 'Psl', 'sea_level_pressure', 'Sea_Level_Pressure']

    data = list(ds.keys())

    for data_name in data:

        for slp_name in slp_names:
            if (data_name == slp_name):
                print(data_name, slp_name)
                ds = ds.rename({data_name: slp})


    # Covert 0-360°E to -180 - 180°E
    if lon180:
        ds['lon'] = ds.lon.where(ds.lon < 180, ds.lon - 360)

    return ds
