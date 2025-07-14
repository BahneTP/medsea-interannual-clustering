import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker


# def plot_anomalies(ds, feature: str, cmin, cmax, title, depths_to_plot):
#     import matplotlib.pyplot as plt

#     hov = ds[feature]
#     hov_mean = hov.mean(dim=['latitude', 'longitude'])

#     dayofyear = hov_mean['time'].dt.dayofyear
#     climatology = hov_mean.groupby(dayofyear).mean(dim='time')
#     climatology_std = hov_mean.groupby(dayofyear).std(dim='time')

#     clim = climatology.sel(dayofyear=dayofyear)
#     std = climatology_std.sel(dayofyear=dayofyear)
#     anomaly = (hov_mean - clim) / std

#     fig, ax = plt.subplots(figsize=(8, 5))

#     p = anomaly.plot(
#         ax=ax,
#         x='time',
#         y='depth',
#         cmap='coolwarm',
#         yincrease=False,
#         vmin=cmin,
#         vmax=cmax,
#         add_colorbar=False
#     )

#     fig.subplots_adjust(bottom=0.18)
#     cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
#     cbar = fig.colorbar(p, cax=cax, orientation="horizontal", pad=0.2)
#     cbar.set_label(title, fontsize=12)

#     ax.set_xlabel('Year', fontsize=12)
#     ax.set_ylabel('Depth (m)', fontsize=12)

#     ax2 = ax.twinx()

#     for depth in depths_to_plot:

#         profile_anom = anomaly.sel(depth=depth, method='nearest')
#         profile_smooth = profile_anom.rolling(time=365, center=True).mean()

#         ax2.plot(
#             profile_smooth['time'],
#             profile_smooth,
#             label=f'{depth:.1f} m',
#             color='black',
#             linewidth=1.2
#         )
#         ax2.set_ylim(-2, 2)

#     ax2.legend(loc='upper right')

#     plt.title('')
#     plt.show()


def plot_anomalies(ds, feature: str, cmin, cmax, title, depths_to_plot, smoothing_factor, mld, feature_graphs):
    import matplotlib.pyplot as plt

    anomalys = []
    for feature in [feature] + [x for x,_ in feature_graphs]:

        hov = ds[feature]
        hov_mean = hov.mean(dim=['latitude', 'longitude'])

        dayofyear = hov_mean['time'].dt.dayofyear
        climatology = hov_mean.groupby(dayofyear).mean(dim='time')
        climatology_std = hov_mean.groupby(dayofyear).std(dim='time')

        clim = climatology.sel(dayofyear=dayofyear)
        std = climatology_std.sel(dayofyear=dayofyear)
        anomaly = (hov_mean - clim) / std
        anomalys.append(anomaly)

    fig, ax = plt.subplots(figsize=(8, 5))

    p = anomalys[0].plot(
        ax=ax,
        x='time',
        y='depth',
        cmap='coolwarm',
        yincrease=False,
        vmin=cmin,
        vmax=cmax,
        add_colorbar=False
    )
    if mld:
        
        mld = ds["mlotst"].mean(dim=['latitude', 'longitude']).sel(depth=75, method='nearest')
        f_smooth = mld.rolling(time=smoothing_factor, center=True).mean()


        ax.plot(
            f_smooth['time'],
            f_smooth,
            label="Mixed Layer Depth",
            color="Brown",
            linewidth=1.4
        )
        ax.legend(loc='upper left')
    

    fig.subplots_adjust(bottom=0.18)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    cbar = fig.colorbar(p, cax=cax, orientation="horizontal", pad=0.2)
    cbar.set_label(title, fontsize=12)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)

    ax2 = ax.twinx()

    # Colormap wählen, etwa 'tab10' für maximal 10 gut unterscheidbare Farben
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(feature_graphs))]
    if len(feature_graphs) * len(depths_to_plot) == 2:
        colors = ["black", (0.3,0.3,0.3)]
    if len(feature_graphs) * len(depths_to_plot) == 1:
        colors = ["black"]

    for i, feature in enumerate([x for x, _ in feature_graphs]):
        for j, depth in enumerate(depths_to_plot):
            profile_anom = anomalys[i+1].sel(depth=depth, method='nearest')
            profile_smooth = profile_anom.rolling(time=smoothing_factor, center=True).mean()

            ax2.plot(
                profile_smooth['time'],
                profile_smooth,
                label=f'{feature_graphs[i][1]} ({depth:.1f} m)',
                color=colors[(i+j)],
                linewidth=1.4
            )
            ax2.set_ylim(-2, 2)

        ax2.legend(loc='upper right')

    plt.title('')
    plt.savefig(title[:-9]+".tiff", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    plt.show()



def plot_hovmoeller(ds, var_name, title, year, contour_levels, mean_space=True, sigma=1.0, anomaly=False, cmin=-1, cmax=-1):
    ds = ds.sortby('depth')
    hov = ds[var_name]
    if mean_space:
        hov = hov.mean(dim=['latitude', 'longitude'])

    if anomaly:
        climatology = hov.groupby('time.dayofyear').mean('time')
        hov = hov.groupby('time.dayofyear') - climatology

    hov_year = hov.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))

    cmin = hov_year.min().item() if cmin == -1 else cmin
    cmax = hov_year.max().item() if cmax == -1 else cmax

    arr = hov_year.values
    arr_smoothed = scipy.ndimage.gaussian_filter(arr, sigma=sigma)

    hov_year_smooth = xr.DataArray(
        arr_smoothed,
        dims=hov_year.dims,
        coords=hov_year.coords
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    pcm = hov_year_smooth.plot(
        ax=ax,
        x='time',
        y='depth',
        cmap='viridis',
        vmin=cmin,
        vmax=cmax,
        yincrease=False,
        cbar_kwargs={'label': f'{var_name} (°C)', 'extend': 'neither'}
    )

    cs = hov_year_smooth.plot.contour(
        ax=ax,
        x='time',
        y='depth',
        levels=contour_levels,
        colors='black',
        linewidths=0.8,
        yincrease=False
    )

    ax.clabel(cs, fmt='%2.0f °C', fontsize=8)

    ax.set_title(f'{year}', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    pcm.colorbar.ax.tick_params(labelsize=10)
    pcm.colorbar.set_label(title, fontsize=12)

    plt.show()



def plot_hovmoeller_grid(
    ds, var_name, years, cmin, cmax, contour_levels, mean_space=True, sigma=1.0,
    nrows=2, ncols=4, anomaly=False, cbartitle="", logscale=False, yearcolor="black"
):
    ds = ds.sortby('depth')
    hov = ds[var_name]
    if mean_space:
        hov = hov.mean(dim=['latitude', 'longitude'])

    if anomaly:
        climatology = hov.groupby('time.dayofyear').mean('time')
        hov = hov.groupby('time.dayofyear') - climatology

    vmin = hov.min().item() if cmin == -1 else cmin
    vmax = hov.max().item() if cmax == -1 else cmax

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.5 * ncols, 2* nrows),
        sharex=False,
        sharey=True
    )

    axes = axes.flatten()
    pcolormesh = None

    for i, (ax, year) in enumerate(zip(axes, years)):
        hov_year = hov.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        arr = hov_year.values
        arr_smoothed = scipy.ndimage.gaussian_filter(arr, sigma=sigma)
        hov_year_smooth = xr.DataArray(
            arr_smoothed,
            dims=hov_year.dims,
            coords=hov_year.coords
        )

        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if logscale else None

        pcolormesh = hov_year_smooth.plot(
            ax=ax,
            x='time',
            y='depth',
            cmap='viridis',
            yincrease=False,
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax,
            norm=norm
        )

        cs = hov_year_smooth.plot.contour(
            ax=ax,
            x='time',
            y='depth',
            levels=contour_levels,
            colors='black',
            linewidths=0.6,
            yincrease=False
        )

        ax.clabel(cs, fontsize=8)
        ax.text(
            0.03,
            0.03,
            f'{year}',
            ha='left',
            va='bottom',
            fontsize=10,
            color=yearcolor,
            transform=ax.transAxes,
        )

        row = i // ncols
        col = i % ncols

        if col == 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('')

        if row != nrows - 1:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('')
            xticklabels = [item.get_text() for item in ax.get_xticklabels()]
            if xticklabels:
                xticklabels[0] = ''
            ax.set_xticklabels(xticklabels)

    for ax in axes[len(years):]:
        ax.set_visible(False)

    cbar_ax = fig.add_axes([0.2, 0, 0.6, 0.02])
    cbar = fig.colorbar(
        pcolormesh,
        cax=cbar_ax,
        orientation='horizontal'
    )
    cbar.set_label(cbartitle, fontsize=10)

    if logscale:
        ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
        formatter = mticker.LogFormatter(base=10, labelOnlyBase=False)
    else:
        ticks = np.linspace(vmin, vmax, num=5)
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))  # Erzwingt sci-notation in deinem Bereich
        formatter.set_useOffset(False)


    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.tick_params(labelsize=10)

    fig.text(
        0.02, 0.5, 'Depth (m)', va='center', ha='center', rotation='vertical', fontsize=10
    )

    fig.text(
        0.5, 0.04, 'Month', va='center', ha='center', fontsize=10
    )
    plt.subplots_adjust(
        left=0.1,
        right=0.95,
        top=0.95,
        bottom=0.08,
        wspace=0.05,
        hspace=0.05
    )

    plt.show()



def plot_variables_overyear(mlotst, title: str, ylabel: str, anomaly=False):

    fig, ax = plt.subplots(figsize=(8, 4))
    
    if anomaly:
        doy = mlotst['time'].dt.dayofyear
        mean_doy = mlotst.groupby(doy).mean(dim='time')
        std_doy = mlotst.groupby(doy).std(dim='time')

        mean_for_time = mean_doy.sel(dayofyear=doy)
        std_for_time = std_doy.sel(dayofyear=doy)

        anomalies = (mlotst - mean_for_time) / std_for_time
        ax.plot(mlotst['time'], anomalies)
    else:
        ax.plot(mlotst['time'], mlotst)

    # ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.8)
    ax.grid(True)
    ax.legend()
    plt.show()


def plot_variables_overyear_selected_months_multiline(data, title: str, ylabel: str, months: list, y1: int, y2: int):

    month_dict = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    for month in months:
        data_month = data.sel(time=data['time.month'] == month)
        data_month_mean = data_month.groupby('time.year').mean('time')
        ax.plot(data_month_mean['year'], data_month_mean, label=month_dict[month])


    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.8)
    ax.grid(True)
    ax.legend(title='')
    ax.set_ylim(y1, y2)
    plt.show()
