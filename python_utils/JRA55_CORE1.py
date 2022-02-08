import xarray as xr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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


def multiple_linear_regression_reanalysis(dependent_var, independent_var1, independent_var2, var_names=['uas', 'vas']):
    ''''''
    # process dependent data
    dep_var = dependent_var.values

    array_size = independent_var1[var_names[0]].shape
    for ii in tqdm(range(array_size[1])):
        for jj in range(array_size[2]):

            # cut single grid point
            temp_indep_var1 = independent_var1[var_names[0]].values[:, ii, jj]
            temp_indep_var2 = independent_var2[var_names[1]].values[:, ii, jj]

            # bring the independent data into the right shape
            indep_vars = np.empty((len(temp_indep_var1), 2))
            for i in range(len(temp_indep_var1)):
                indep_vars[i, 0] = temp_indep_var1[i]
                indep_vars[i, 1] = temp_indep_var2[i]

            # initialize the final arrays in the very first loop
            if (ii == 0) & (jj == 0):
                R2_prediction_full = np.empty((array_size[1], array_size[2]))
                intercepts_full = np.empty((array_size[1], array_size[2]))
                predicted_var_full = np.empty((len(dep_var), array_size[1], array_size[2]))
                coeffs_full = np.empty((2, array_size[1], array_size[2]))

            ##### perform regression fit for whole data
            lr = LinearRegression()
            lr.fit(indep_vars, dep_var)
            # compute R^2 between
            R2_prediction_full[ii, jj] = lr.score(indep_vars, dep_var)
            predicted_var_full[:, ii, jj] = lr.predict(indep_vars)
            intercepts_full[ii, jj] = lr.intercept_
            coeffs_full[:, ii, jj] = lr.coef_

    return predicted_var_full, R2_prediction_full, coeffs_full, intercepts_full, predicted_var_full


def linear_regression_reanalysis(dependent_var, independent_var1, var_names='psl'):
    ''''''
    # process dependent data
    dep_var = dependent_var.values

    array_size = independent_var1[var_names].shape
    for ii in tqdm(range(array_size[1])):
        for jj in range(array_size[2]):

            # cut single grid point
            temp_indep_var1 = independent_var1[var_names].values[:, ii, jj]

            # bring the independent data into the right shape
            indep_vars = np.empty((len(temp_indep_var1), 1))
            for i in range(len(temp_indep_var1)):
                indep_vars[i, 0] = temp_indep_var1[i]

            # initialize the final arrays in the very first loop
            if (ii == 0) & (jj == 0):
                R2_prediction_full = np.empty((array_size[1], array_size[2]))
                intercepts_full = np.empty((array_size[1], array_size[2]))
                predicted_var_full = np.empty((len(dep_var), array_size[1], array_size[2]))
                coeffs_full = np.empty((2, array_size[1], array_size[2]))

            ##### perform regression fit for whole data
            lr = LinearRegression()
            lr.fit(indep_vars, dep_var)
            # compute R^2 between
            R2_prediction_full[ii, jj] = lr.score(indep_vars, dep_var)
            predicted_var_full[:, ii, jj] = lr.predict(indep_vars)
            intercepts_full[ii, jj] = lr.intercept_
            coeffs_full[:, ii, jj] = lr.coef_

    return predicted_var_full, R2_prediction_full, coeffs_full, intercepts_full


def CorrelationReanalysis(var, field, variable='psl'):
    field = field[variable].values

    array_size = field.shape
    # preallocate array
    corrcoeffs = np.empty((array_size[1], array_size[2]))
    p_values = np.empty((array_size[1], array_size[2]))

    for ii in tqdm(range(array_size[1])):
        for jj in range(array_size[2]):
            corrcoeffs[ii, jj], p_values[ii, jj] = pearsonr(var, field[:, ii, jj])

    return corrcoeffs, p_values


def CorrCoeffRegressOri(p_rea_full, vol_transport):
    '''
    Computes the Pearson Correlation Coefficient between the reconstructed timeseries (from regression) and the original timeseries
    '''

    # preallocate arrays
    array_shape = p_rea_full.shape
    corrcoeff = np.empty((array_shape[1], array_shape[2]))
    p_val = np.empty((array_shape[1], array_shape[2]))

    # compute correlation coeff.
    for ii in tqdm(range(array_shape[1])):
        for jj in range(array_shape[2]):
            corrcoeff[ii, jj], p_val[ii, jj] = pearsonr(vol_transport.values, p_rea_full[:, ii, jj])

    return corrcoeff, p_val


def PreferredWindDir(c_rea, corrcoeff, p_val, min_correlation=.5, p_val_max=.05):
    '''
    Computes the Preferred Wind Direction of a multi parameter (u,v) wind field regression
    '''

    # Compute the vectors
    wind_vectors = c_rea / np.sqrt(c_rea[0, :, :] ** 2 + c_rea[1, :, :] ** 2)[np.newaxis, :]

    # Optional mask
    mask = np.where((corrcoeff < min_correlation) | (p_val > p_val_max), np.nan, 1)
    wind_vectors = wind_vectors * mask[np.newaxis, :]

    return wind_vectors








