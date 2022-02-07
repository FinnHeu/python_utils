import xarray as xr
import pandas as pd
import numpy as np
from eofs.xarray import Eof
from .dataset_operations import select_winter_month, dataset_to_cfconvention

def Dec2MarMeanTransport(srcpath: str, newdates: tuple, Tmin=-5):
    '''

    :return:
    '''

    ds = xr.open_dataset(srcpath)

    # Shift by one month to allow groupby year (winter)
    newtime = pd.date_range(newdates[0], newdates[-1], freq='m')
    
    if len(newtime) != len(ds.time):
        raise ValueError('The length of the time vectors does not match!')
    
    ds['time'] = newtime

    # Select winter month
    ds = select_winter_month(ds, month=[1,2,3,4])

    # Apply temperature filter and accumulate transport
    transport = ds.transport_across.where(ds.temp > Tmin, 0).sum(dim=['elem','nz1'])

    # Groupby year
    transport = transport.groupby('time.year').mean()

    return ds, transport


def NAOindex(src_path: str, newdates: tuple, slpvar='psl', timeslice=('1960','2020')):
    '''


    '''
    ds = xr.open_dataset(src_path)[slpvar]

    # -180:180°E
    if any([l > 180 for l in ds.lon]):
        ds['lon'] = ds.lon.where(ds.lon < 180, ds.lon - 360)
        ds = ds.sortby(ds.lon)

    # restrict region
    ds = ds.sel(lon=slice(-90, 40),
                lat=slice(20, 80),
                time=slice(timeslice[0],
                           timeslice[-1])
                )

    # shift by one month to allow groupby year
    newtime = pd.date_range(newdates[0], newdates[-1], freq='m')

    if len(newtime) != len(ds.time):
        raise ValueError('The length of the time vectors does not match!')

    ds['time'] = newtime

    # select winter month
    ds = select_winter_month(ds, month=[1, 2, 3, 4])

    # Froupby year
    ds = ds.groupby('time.year').mean()
    ds = ds.rename({'year': 'time'})

    # latitude weight
    lat_weight = np.cos(ds.lat * np.pi / 180)

    # transpose
    ds = ds.transpose('time', 'lon', 'lat')

    # EOF
    solver = Eof(ds, weights=lat_weight)
    eofs = solver.eofs(neofs=5, eofscaling=1)
    pcs = solver.pcs(npcs=5, pcscaling=1)
    
    return eofs, pcs










