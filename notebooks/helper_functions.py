import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import torch
import xarray as xr


def plot_cluster_timeline(z, labels):
    """
    z: stacked feature array.
    labels: Output of K-means.

    Plots a Step-Plot of each timesteps cluster.
    """
    # print(z.time.values[:12])
    # print("Anzahl Zeitpunkte:", len(z.time))
    # print(labels[:12])


    time_index = z.time.values  # → this will be the months/timestamps
    
    plt.figure(figsize=(10, 3))
    plt.plot(time_index, labels, drawstyle='steps-mid', color="black")
    
    plt.yticks(np.arange(labels.min(), labels.max() + 1))
    plt.xlabel("Time")
    plt.xlim(time_index[0], time_index[-1])
    plt.ylabel("Cluster ID")
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # jedes Jahr
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()



def plot_average_cluster(z, labels, cmin = None, cmax = None):
    """
    z: stacked feature vector.
    labels: Output of K-Means.
    cmin: First value of the colorbar.
    cmax: Last value of the colorbar.

    Plots a figure containing the average maps of each cluster.
    """
    
    z_stack = z.assign_coords(cluster=("time", labels)) # Getting back to lat / lon.
    z_stack.drop_duplicates('location')
    z_unstacked = z_stack.unstack("location")  # → dims: time, latitude, longitude
    
    # Compute average z-score map per cluster
    cluster_maps = {}
    for cluster_id in np.unique(labels):
        avg_map = z_unstacked.where(z_unstacked.cluster == cluster_id).mean("time")
        cluster_maps[cluster_id] = avg_map
    
    # Sort clusters by ID (to ensure order 0→5 or 1→6)
    sorted_ids = sorted(cluster_maps.keys())
    cluster_maps_ordered = [cluster_maps[i] for i in sorted_ids]
    
    # Determine global color scale range
    vmin = cmin if cmin != None else min([m.min().item() for m in cluster_maps_ordered])
    vmax = cmax if cmax != None else max([m.max().item() for m in cluster_maps_ordered])
    
    # Grid layout
    n_clusters = len(cluster_maps_ordered)
    ncols = 3
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.5 * nrows),
                             subplot_kw={'projection': ccrs.Mercator()})
    axes = axes.flatten()
    
    # Plot each cluster
    for i, (ax, data_cluster) in enumerate(zip(axes, cluster_maps_ordered)):
        im = data_cluster.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False
        )
        ax.coastlines()
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    
        # Label with cluster ID (1-based)
        ax.text(
            0.03, 0.03, str(sorted_ids[i]),  # from 1 to 6
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color='black',
            ha='left',
            va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.6)
        )
    
    # Hide empty subplots
    for ax in axes[len(cluster_maps_ordered):]:
        ax.set_visible(False)
    
    # One shared colorbar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Z-score")
    
    # Adjust layout
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, top=0.98, bottom=0.12)
    plt.show()




def plot_multiple_reconstructions(device, X, model, z_stack, indices, title_prefix="Sample"):
    """
    X: numpy array of shape (n_samples, n_features)
    model: trained autoencoder
    z_stack: xarray with dims ("time", "location")
    indices: list of time indices to plot (e.g., [0, 1, 2])
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n), subplot_kw={'projection': ccrs.Mercator()})
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, t_idx in enumerate(indices):
        original_vector = X[t_idx]
        with torch.no_grad():
            reconstructed_vector = model(torch.tensor(original_vector).float().to(device)).cpu().numpy()

        def to_map(vec):
            da = xr.DataArray(vec, coords={"location": z_stack.location}, dims=["location"])
            return da.unstack("location")

        original_map = to_map(original_vector)
        reco_map = to_map(reconstructed_vector)

        for ax, data, title in zip(axes[i], [original_map, reco_map], ["Original", "Reconstruction"]):
            im = data.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap="coolwarm",
                add_colorbar=False
            )
            ax.coastlines()
            ax.set_title(f"{title_prefix} {t_idx}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])

    # Gemeinsame Farbskala
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Z-score")

    plt.tight_layout()
    plt.show()

