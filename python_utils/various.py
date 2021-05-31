import numpy as np

def slp_to_geowind(slp, lon, lat, rho = 1.225, Re = 6371000, f='f-plane', f_central=None):

    '''
    slp_to_geowind.py

    Computes geostrophic winds from sea level pressure field

    Inputs
    ------
    slp: np.ndarray, sea level pressure field
    lon: np.ndarray, longitude vector
    lat: np.ndarray, latitude vector
    rho = 1.225, air density
    Re = 6371000, radius earth
    f='f-plane' or 'real_world'
    f_central=None, central latitude for f-plane

    Returns
    -------
    u, v: zonal and meridional wind components
    '''
    
    # Compute the distances between the gridpoints for computing the gradient
    dist_x = np.diff(lon)[0] * (2 * np.pi * Re / 360) * np.cos(lat/360 * 2 * np.pi)
    dist_y = np.diff(lat) * (2 * np.pi * Re / 360)

    # Compute the gradient
    dpdx = np.ones_like(slp) * np.nan
    dpdy = np.ones_like(slp) * np.nan

    for i in range(len(lat)): dpdx[i,:] = np.gradient(slp[i,:], dist_x[i])
    for i in range(len(lon)): dpdy[:,i] = np.gradient(slp[:,i], np.mean(dist_y))

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

    return u, v
