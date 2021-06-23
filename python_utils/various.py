import numpy as np

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
