import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from eofs.xarray import Eof
from .dataset_operations import select_winter_month


# Subfunctions
def RestrictRegionTime(ds, extent, time1, time2):
    '''
    Cut data to region and time slice
    '''

    # -180:180°E
    if any([l > 180 for l in ds.lon]):
        ds['lon'] = ds.lon.where(ds.lon < 180, ds.lon - 360)
        ds = ds.sortby(ds.lon)

    # restrict region
    ds = ds.sel(lon=slice(extent[0], extent[1]),
                lat=slice(extent[2], extent[3]),
                time=slice(time1,
                           time2)
                )

    return ds

def EofAreaWeighted(ds):
    '''
    Compute the area weighted EOF
    '''

    # select winter month
    ds = select_winter_month(ds, month=[1, 2, 3, 4])

    # Groupby year
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

def TimeShiftForWinterMean(ds, newdates):
    '''
    Shifts the time by one month forward to allow xarray groupby time.year method
    This should be automated later
    '''
    # shift by one month to allow groupby year
    newtime = pd.date_range(newdates[0], newdates[-1], freq='m')

    if len(newtime) != len(ds.time):
        raise ValueError('The length of the time vectors does not match!')

    ds['time'] = newtime

    return ds

def PlotPatternIndex(eof, pc, mode):
    '''
    Plot the results of the computation as map
    '''

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    # PC
    ax[0].bar(pc.time, pc.isel(mode=mode))
    # EOF
    cb = ax[1].contourf(eof.lon, eof.lat, eof.isel(mode=mode)transpose(), cmap='RdBu_r')

    plt.colorbar(cb, ax=ax)

    return

# Index
def NAOindex(src_path: str, newdates: tuple, slpvar='psl', timeslice=('1960', '2020')):
    '''
    Computes the NAO index for given monthly mean SLP data as first area weighted EOF within 90°E - 40°W, 20°N - 80°N
    '''
    extent = (-90, 40, 20, 80)
    # Open Files
    ds = xr.open_dataset(src_path)[slpvar]

    # Restrict region for EOF analysis
    ds = RestrictRegionTime(ds, extent, timeslice[0], timeslice[-1])

    # Shift time for winter means
    ds = TimeShiftForWinterMean(ds, newdates)

    # Apply area weighted EOF
    eofs, pcs = EofAreaWeighted(ds)

    # Plot
    PlotPatternIndex(eofs, pcs, 0)

    return eofs.isel(mode=0), pcs.isel(mode=0)

def BOindex(src_path: str, newdates: tuple, slpvar='psl', timeslice=('1960', '2020')):
    '''
    Computes the NAO index for given monthly mean SLP data as first area weighted EOF within 90°E - 40°W, 20°N - 80°N
    https://doi.org/10.1002/grl.50551
    '''

    extent = (-90, 90, 30, 90)

    # Open Files
    ds = xr.open_dataset(src_path)[slpvar]

    # Restrict region for EOF analysis
    ds = RestrictRegionTime(ds, extent, timeslice[0], timeslice[-1])

    # Shift time for winter means
    ds = TimeShiftForWinterMean(ds, newdates)

    # Apply area weighted EOF
    eofs, pcs = EofAreaWeighted(ds)

    # Plot
    PlotPatternIndex(eofs, pcs, 1)

    return eofs.isel(mode=1), pcs.isel(mode=1)

def NAMindex(src_path: str, newdates: tuple, slpvar='psl', timeslice=('1960', '2020')):
    '''
    Computes the NAM index for given monthly mean SLP data as first area weighted EOF north of 20°N
    https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.loading.shtml
    '''

    extent = (-180, 180, 20, 90)

    # Open Files
    ds = xr.open_dataset(src_path)[slpvar]

    # Restrict region for EOF analysis
    ds = RestrictRegionTime(ds, extent, timeslice[0], timeslice[-1])

    # Shift time for winter means
    ds = TimeShiftForWinterMean(ds, newdates)

    # Apply area weighted EOF
    eofs, pcs = EofAreaWeighted(ds)

    # Plot
    PlotPatternIndex(eofs, pcs, 0)

    return eofs.isel(mode=0), pcs.isel(mode=0)

def ADindex(src_path: str, newdates: tuple, slpvar='psl', timeslice=('1960', '2020')):
    '''
    Computes the NAO index for given monthly mean SLP data as second area weighted EOF north of 70°N°
    https://doi.org/10.1175/JCLI3619.1
    '''

    extent = (-180, 180, 70, 90)

    # Open Files
    ds = xr.open_dataset(src_path)[slpvar]

    # Restrict region for EOF analysis
    ds = RestrictRegionTime(ds, extent, timeslice[0], timeslice[-1])

    # Shift time for winter means
    ds = TimeShiftForWinterMean(ds, newdates)

    # Apply area weighted EOF
    eofs, pcs = EofAreaWeighted(ds)

    # Plot
    PlotPatternIndex(eofs, pcs, 1)

    return eofs.isel(mode=1), pcs.isel(mode=1)