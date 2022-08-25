import numpy as np


def ice_area_fesom(ds, mesh, mesh_diag, box=(10, 60, 72, 82)):
    # Cut to box
    inds = np.where((mesh.x2 > box[0]) & (mesh.x2 < box[1]) & (mesh.y2 > box[2]) & (mesh.y2 < box[3]))[0]
    ds_cropped = ds.isel(nod2=inds)
    mesh_diag_cropped = mesh_diag.isel(nod2=inds)

    # create sea ice mask
    ds_cropped['is_ice'] = ds_cropped.a_ice * 0 + np.where(ds_cropped.a_ice > .15, 1, 0).astype('float32')

    # sum over non-masked nodal areas
    ds_cropped['ice_area'] = (
    ('time'), (ds_cropped.is_ice * mesh_diag_cropped.nod_area.isel(nz=0)).sum(dim='nod2').values)

    return ds_cropped


def ice_area_nsidc(ds, box=(10, 60, 72, 82)):
    # Cut to box
    ds = ds.sortby('latitude')
    ds_cropped = ds.sel(latitude=slice(box[2], box[3]), longitude=slice(box[0], box[1]))

    # Expand Gridcell Area array
    ds_cropped['Gridcell_Area_ext'] = ds_cropped.Gridcell_Area * (ds_cropped.seaice_conc.fillna(0) * 0 + 1)

    # Create sea-ice mask
    ds_cropped['is_ice'] = ds_cropped.seaice_conc * 0 + np.where(
        (ds_cropped.seaice_conc > 15) & (ds_cropped.seaice_conc <= 100), 1, 0).astype('float32')

    # sum over non masked gridcell areas
    ds_cropped['ice_area'] = (ds_cropped.is_ice * ds_cropped.Gridcell_Area_ext).sum(dim=('longitude', 'latitude'))

    return ds_cropped
