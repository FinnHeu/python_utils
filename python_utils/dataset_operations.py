import numpy as np
import xarray as xr
import glob


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

    time_names = ['TIME', 'Time']
    lon_names = ['LONGITUDE', 'Longitude', 'longitude', 'LON', 'Lon', 'LONS', 'lons', 'Lons']
    lat_names = ['LATITUDE', 'Latitude', 'latitude', 'LAT', 'Lat', 'LATS', 'lats', 'Lats']

    coords = list(ds.coords)
    for coord_name in coords:

        for time_name in time_names:
            if (coord_name == time_name):
                ds = ds.rename({coord_name: 'time'})

        for lon_name in lon_names:
            if (coord_name == lon_name):
                ds = ds.rename({coord_name: 'longitude'})

        for lat_name in lat_names:
            if (coord_name == lat_name):
                ds = ds.rename({coord_name: 'latitude'})

    # Rename Data variables

    slp_names = ['SLP', 'Slp', 'PSL', 'psl', 'Psl',
                 'sea_level_pressure', 'Sea_Level_Pressure', 'msl']

    data = list(ds.keys())

    for data_name in data:

        for slp_name in slp_names:
            if (data_name == slp_name):
                print('Renamed: ', slp_name, ' to: slp')
                ds = ds.rename({data_name: 'slp'})

    # Covert 0-360°E to -180 - 180°E
    if lon180:
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)

    # Rename Data variables

    temp_names = ['Temp', 'TEMP', 'tas', 'T_2_MOD', 'T_10_MOD',
                  'temperature', 'Temperature', 'TEMPERATURE']

    data = list(ds.keys())

    for data_name in data:

        for temp_name in temp_names:
            if (data_name == temp_name):
                ds = ds.rename({data_name: temp})

    # Covert 0-360°E to -180 - 180°E
    if lon180:
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)

    # Rearange dimensions
    #ds = ds.transpose('time','lon','lat')

    return ds


def select_winter_month(ds, month=[12, 1, 2, 3, 4, 5], mean=False):
    '''
    select_winter_month.py

    selects winter month from an xarray dataset

    Inputs
    ------
    ds (xr.dataset)
        dataset to process
    month (list)
        months to select
    mean (bool)
        apply time mean

    '''

    # Apply cf conventions if nesseccary
    if not 'time' in list(ds.coords):
        ds = dataset_to_cfconvention(ds)

    groups = ds.groupby('time.month').groups

    winter_inds = list()

    for i in range(len(month)):
        winter_inds += groups[month[i]]

    winter_inds = np.sort(winter_inds)

    ds = ds.isel(time=winter_inds)

    if mean:
        ds = ds.mean(dim='time')

    return ds


def crop_fesom_to_region(src, mesh, dest, region=[0, 90, 50, 90]):
    '''crop_fesom_to_region.py

    Crops fesom output to a specific region to save memory

    Inputs
    ------
    src (str)
        paths to files to process
    mesh (fesom.mesh)
        fesom mesh object
    dest (str)
        destination path to save output
    region (list)
        list of form [lon1 lon2 lat1 lat2] defining the box

    Returns
    -------

    '''
    # handle strings and list of strings
    if isinstance(src, str):
        src = [src]
    elif isinstance(src, list):
        pass
    elif isinstance(src, np.ndarray):
        pass
    else:
        raise ValueError('data type not supported')

    # compute indices to crop
    inds = np.where((mesh.x2 >= region[0]) & (mesh.x2 <= region[1]) & (
        mesh.y2 >= region[2]) & (mesh.y2 <= region[3]))[0]

    # load individual files, crop and save
    for i in tqdm(range(len(src))):

        ds = xr.open_dataset(src[i], chunks={'nod2': 1e4}).isel(nod2=inds).load()
        ds.to_netcdf(dest[i])

    return inds
