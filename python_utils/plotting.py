import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np


def plot_background(ax, extent="BS", land=True, coastline=True, landcolor='lightgrey', tick_labels=True):

    '''
    plot background.py

    Plots background plot and adds cartopy features for specific regions.

    Inputs
    ------
    ax: cartopy geoaxis
    extent: 'BS', 'Arc', 'Arc+', [lon lon, lat lat]
    land=True
    coastline=True

    Returns
    -------
    ax


    Usage
    -------
    fig, ax = plt.subplots(1,1, figsize=(10,7.5), subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
    plot_background(ax, extent='BS', land=False)
    '''


    if extent == "BS":

        gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        x_inline=True,
        y_inline=False,
        zorder=20,
    )
        ax.set_extent([0, 90, 66, 90], crs=ccrs.PlateCarree())

        gl.xlocator = mticker.FixedLocator(
            [20, 30, 40, 50, 60, 80],
        )
        gl.ylocator = mticker.FixedLocator(np.arange(70, 90, 5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    elif extent == "BS_close":

        gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        x_inline=True,
        y_inline=False,
        zorder=20,
    )
        ax.set_extent([17.5, 62.5, 67.5, 82.5], crs=ccrs.PlateCarree())

        gl.xlocator = mticker.FixedLocator(
            [20, 40, 60],
        )
        gl.ylocator = mticker.FixedLocator(np.arange(70, 82.5, 2.5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
    elif extent == "BS_centered":

        gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        x_inline=True,
        y_inline=True,
        zorder=20,
    )
        ax.set_extent([0,80,68,82.5], crs=ccrs.PlateCarree())

        gl.xlocator = mticker.FixedLocator(
            [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80],
        )
        gl.ylocator = mticker.FixedLocator(np.arange(70, 85, 2.5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    elif extent == "Arc+":

        gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        x_inline=False,
        y_inline=True,
        zorder=20,
    )
        ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())

        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(45, 90, 5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    elif extent == "Arc":

        gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        x_inline=False,
        y_inline=True,
        zorder=20,
    )
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(55, 90, 5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    else:
        gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        x_inline=False,
        y_inline=True,
        zorder=20,
    )
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(35, 90, 5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER


    if land:
        ax.add_feature(cfeature.LAND, zorder=10, color=landcolor)
    if coastline:
        ax.add_feature(cfeature.COASTLINE, zorder=10)

    gl.xlabel_style = {"size": 12, "rotation": 0}
    gl.ylabel_style = {"size": 12, "rotation": 0}

    if not tick_labels:
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlabels_bottom = False
        gl.ylabels_right = False

    return ax
