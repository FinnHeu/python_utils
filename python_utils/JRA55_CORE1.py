import xarray as xr
import pandas as pd
from .dataset_operations import select_winter_month, dataset_to_cfconvention

def Dec2MarMeanTransport(srcpath: str, newdates: tuple, Tmin=-5):
    '''

    :return:
    '''

    ds = xr.open_dataset(srcpath)
    ds = dataset_to_cfconvention(ds)

    # Shift by one month to allow groupby year (winter)
    newtime = pd.date_range(newdates[0], newdates[-1], freq='m')
    if len(newtime) != len(ds.time):
        raise ValueError('The length of the time vectors does not match!')

    # Select winter month
    ds = select_winter_month(ds, month=[1,2,3,4])

    # Apply temperature filter and accumulate transport
    transport = ds.transport_across.where(ds.temp > Tmin, 0).sum(dim=['elem','nz1'])

    # Groupby year
    transport = transport.groupby('time.year').mean()

    return ds, transport





