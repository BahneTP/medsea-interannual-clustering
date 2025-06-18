import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import torch
import xarray as xr
from sklearn.cluster import KMeans

def preprocessing(ds, features: list, depths: list, dim):
    """
    ds: Dataset in form of an xarray object.
    feature: List of features to combine. "sol" for Salinity or "temperature for Temperature.
    depth: "Any value from 0 to 1062, but it will assign the closest existing."

    returns a stacked array of the "feature".
    """

    def standardize(group):
        m = group.mean("time")
        s = group.std("time")
        return (group - m) / s

    feature_vectors = []
    for feature in features:

        for depth in depths:
            data = ds[feature].sel(depth=depth, method="nearest")
            data = data.assign_coords(month=data["time"].dt.month)
        
            z = data.groupby("month").apply(standardize)
            z = z.stack(location=("latitude", "longitude"))
            feature_vectors.append(z)
    
    z_concat = xr.concat(feature_vectors, dim=dim)
    z_concat = z_concat.dropna(dim="location", how="any")

    return z_concat  # → xarray.DataArray mit dims: ("time", "location")

def apply_kmeans(X, k: int):
    """Simply applies the k-means algorithm with "k" clusters."""

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
    return kmeans.fit_predict(X)

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


def plot_cluster_map(z_stack, labels):
    """
    z_stack: stacked feature array.
    labels: output from k-means.

    Plots a map which shows the clusters regions.
    """
    
    cluster_da = xr.DataArray(labels, coords=[z_stack.location], dims=["location"])
    cluster_map = cluster_da.unstack("location")

    # Plotting
    lat = cluster_map.latitude
    lon = cluster_map.longitude

    fig = plt.figure(figsize=(7, 6))
    ax = plt.axes(projection=ccrs.Mercator())

    mesh = ax.pcolormesh(
        lon, lat, cluster_map.values,
        cmap="tab10",
        transform=ccrs.PlateCarree()
    )

    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Legend
    for cluster_id in np.unique(labels):
        plt.scatter([], [], c=[plt.cm.tab10(cluster_id)], label=f"Cluster {cluster_id}")
    plt.legend(
        title="",
        loc="lower left",
        bbox_to_anchor=(0.15, -0.02))

    plt.tight_layout()
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


def reconstructed_to_stack(ds, feature: str, depth: float, ae_recons: list):
    """
    ds: Xarray object.
    feature: Either "thetao" for Temperature or "so" for Salinity.
    depth: int.
    ae_recons: List of the autoencoders reconstructions.
    
    Builds an xarray DataArray with dimensions ("time", "location") for a given feature and depth,
    based on a list of Autoencoder reconstructions.
    """

    # Size of the features and depths in the reconstruction-vector.
    sizes = {
        "thetao": {
            47.37: 40657,
            318.13: 31973,
            1062.44: 24517,
        },
        "so": {
            47.37: 40657,
            318.13: 31973,
            1062.44: 24517,
        }
    }

    # Set depth.
    depth = ds[feature].sel(depth=depth, method="nearest").depth.values.item()
    depth = round(depth, 2)


    # Calculate Indices.
    def compute_start_index(feature, depth):
        order = [("thetao", 47.37), ("thetao", 318.13), ("thetao", 1062.44),
                 ("so", 47.37), ("so", 318.13), ("so", 1062.44)]
        start = 0
        for f, d in order:
            if f == feature and d == depth:
                return start
            start += sizes[f][d]
        raise ValueError(f"Feature {feature} with {depth} not found.")

    start_idx = compute_start_index(feature, depth)
    n_points = sizes[feature][depth]

    # Ensure the shape of the reconstructions is correct.
    for recon in ae_recons:
        if recon.shape[0] < start_idx + n_points:
            raise ValueError(f"One reconstruction-vector is too short for startindex + length.")

    # Build array: time × location
    recon_values = [recon[start_idx:start_idx + n_points] for recon in ae_recons]
    recon_array = np.stack(recon_values, axis=0)  # → shape: (time, location)

    # Geet coordinates.
    base = ds[feature].sel(depth=depth, method="nearest")
    base = base.stack(location=("latitude", "longitude"))
    base = base.dropna(dim="location", how="any")
    coords = base["location"]

    # Extracting the Time-Coordinate out of the dataframe.
    time_coords = ds["time"]

    # Building the DataArray.
    out = xr.DataArray(
        data=recon_array,
        dims=("time", "location"),
        coords={"time": time_coords, "location": coords}
    )

    return out
