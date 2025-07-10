import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import torch
import xarray as xr
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment

def interpolate_time_linear(data: xr.DataArray, factor: int = 2) -> xr.DataArray:
    """
    Interpoliert ein DataArray über die Zeitachse und erzeugt künstlich neue Zeitpunkte.

    Parameters:
    - data: Das ursprüngliche xarray.DataArray mit Dimension "time"
    - factor: Wie viele zusätzliche Zeitpunkte? z. B. 2 = doppelte Anzahl

    Returns:
    - Ein DataArray mit interpolierten Werten und verfeinerter Zeitauflösung
    """

    time_orig = data.time
    n = len(time_orig)

    t_start = time_orig[0].values.astype("datetime64[ns]").astype("int64")
    t_end = time_orig[-1].values.astype("datetime64[ns]").astype("int64")
    t_interp = np.linspace(t_start, t_end, factor * n)
    new_time = pd.to_datetime(t_interp.astype("int64"))

    return data.interp(time=("time", new_time))


def preprocessing(ds, features: list, depths: list, dim: str, trend_removal: bool, interpolate: int):
    """
    feature: List of features to combine. "sol" for Salinity or "temperature for Temperature.
    depth: "Any value from 0 to 1062, but it will assign the closest existing."

    returns a stacked array of the "feature".
    """

    def standardize(group):
        m = group.mean("time")
        s = group.std("time")
        return (group - m) / s
        
    # Only masking the atlantic.
    # mask = ~((lon2d < 0) & (lat2d > 41))

    lon2d, lat2d = xr.broadcast(ds.longitude, ds.latitude)
    atlantic_mask = ~((lon2d < 0) & (lat2d > 41))
    blacksea_mask = ~((lon2d > 27) & (lat2d > 41))
    mask = atlantic_mask & blacksea_mask
    
    feature_vectors = []
    for feature in features:
        for depth in depths:
            data = ds[feature].sel(depth=depth, method="nearest")
            data = data.where(mask)
    
            if interpolate and interpolate > 1:
                data = interpolate_time_linear(data, factor=interpolate)

            if trend_removal:
                fit = data.polyfit(dim="time", deg=1)
                trend = xr.polyval(coord=data['time'], coeffs=fit.polyfit_coefficients)
                data = data - trend
            
            # Monat zuweisen für Gruppierung
            data = data.assign_coords(month=data["time"].dt.month)
    
            # Standardisierung
            z = data.groupby("month").apply(standardize)
            z = z.stack(location=("latitude", "longitude"))
            feature_vectors.append(z)

    
    z_concat = xr.concat(feature_vectors, dim=dim)
    z_concat = z_concat.dropna(dim="location", how="any")

    return z_concat  # → xarray.DataArray mit dims: ("time", "location")


def preprocessing_conv(ds, features: list, depths: list, trend_removal: bool, interpolate: int):
    def standardize(group):
        m = group.mean("time")
        s = group.std("time")
        return (group - m) / s

    lon2d, lat2d = xr.broadcast(ds.longitude, ds.latitude)
    atlantic_mask = ~((lon2d < 0) & (lat2d > 41))
    blacksea_mask = ~((lon2d > 27) & (lat2d > 41))
    geo_mask = atlantic_mask & blacksea_mask
    
    channels = []
    masks = []

    for feature in features:
        for depth in depths:
            try:
                data = ds[feature].sel(depth=depth, method="nearest")
                data = data.where(geo_mask)

                if interpolate and interpolate > 1:
                    data = interpolate_time_linear(data, factor=interpolate)
                    
                if trend_removal:
                    fit = data.polyfit(dim="time", deg=1)
                    trend = xr.polyval(coord=data['time'], coeffs=fit.polyfit_coefficients)
                    data = data - trend
                
                data = data.assign_coords(month=data["time"].dt.month)
                z = data.groupby("month").apply(standardize)
                z = z.transpose("time", "latitude", "longitude")

                nan_mask = (~np.isnan(z)).astype(np.float32)
                z_filled = z.fillna(0.0)

                channels.append(z_filled)
                masks.append(nan_mask)

            except Exception as e:
                print(f"⚠️ Skipping {feature}@{depth}m: {e}")
                continue

    z_all = xr.concat(channels, dim="channel").transpose("time", "channel", "latitude", "longitude")
    m_all = xr.concat(masks, dim="channel").transpose("time", "channel", "latitude", "longitude")

    return z_all.values.astype(np.float32), m_all.values.astype(np.float32)


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



def plot_average_cluster(z, labels, cmin = None, cmax = None, ncols=3):
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
        avg_map = avg_map.sortby(["latitude", "longitude"])
        cluster_maps[cluster_id] = avg_map
    
    # Sort clusters by ID (to ensure order 0→5 or 1→6)
    sorted_ids = sorted(cluster_maps.keys())
    cluster_maps_ordered = [cluster_maps[i] for i in sorted_ids]
    
    # Determine global color scale range
    vmin = cmin if cmin != None else min([m.min().item() for m in cluster_maps_ordered])
    vmax = cmax if cmax != None else max([m.max().item() for m in cluster_maps_ordered])
    
    # Grid layout
    n_clusters = len(cluster_maps_ordered)
    # ncols = 3
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
    cluster_map = cluster_da.unstack("location").sortby(["latitude", "longitude"])


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


# ---------- Autoencoder-methods ----------


def reconstruct_in_batches(X, model,  device, batch_size=16):
    model.eval()
    recon_list = []

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).to(device))
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for (x_batch,) in loader:

            x_batch = x_batch.to(device)
            try:
                x_recon , _, _= model(x_batch)
            except:
                x_recon = model(x_batch)
            recon_list.append(x_recon.cpu())

    return torch.cat(recon_list, dim=0)

def reconstruction_to_vector_masked_positions(X_recon, valid_flat_mask):
    """
    Extracts only valid positions from each CAE-reconstruction (time, C*H*W).
    
    Args:
        X_recon: torch.Tensor or np.ndarray, shape (T, C, H, W)
        valid_flat_mask: 1D boolean array of shape (C * H * W)

    Returns:
        np.ndarray of shape (T, N_valid_positions)
    """
    if isinstance(X_recon, torch.Tensor):
        X_recon = X_recon.detach().cpu().numpy()
    if isinstance(valid_flat_mask, torch.Tensor):
        valid_flat_mask = valid_flat_mask.detach().cpu().numpy()

    T = X_recon.shape[0]
    flat = X_recon.reshape(T, -1)
    return flat[:, valid_flat_mask]


def reconstructed_to_stack(ds, feature: str, depth: float, ae_recons: list, coords):
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
            47.37: 34974,
            318.13: 27736,
            1062.44: 20811,
        },
        "so": {
            47.37: 34974,
            318.13: 27736,
            1062.44: 20811,
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

    # Extracting the Time-Coordinate out of the dataframe.
    time_coords = ds["time"]

    # Building the DataArray.
    out = xr.DataArray(
        data=recon_array,
        dims=("time", "location"),
        coords={"time": time_coords, "location": coords}
    )

    return out

def plot_reconstruction_comparison(z_stack_original: xr.DataArray, 
                                   z_stack_recon: xr.DataArray, 
                                   time_indices: list, 
                                   cmin=None, 
                                   cmax=None):
    """
    Plots Original and Reconstruction of given time-indices.

    z_stack_original: xr.DataArray (time, location), from preprocessing.
    z_stack_recon: xr.DataArray with the same shape.
    time_indices: List of time-indices to plot.
    cmin, cmax: Scale of the colorbar.
    """

    # (time, location) → (time, lat, lon)
    z_unstacked_orig = z_stack_original.unstack("location").sortby(["latitude", "longitude"])
    z_unstacked_recon = z_stack_recon.unstack("location").sortby(["latitude", "longitude"])

    vmin = cmin if cmin is not None else min(z_unstacked_orig.min().item(), z_unstacked_recon.min().item())
    vmax = cmax if cmax is not None else max(z_unstacked_orig.max().item(), z_unstacked_recon.max().item())

    n = len(time_indices)
    fig, axes = plt.subplots(n, 2, figsize=(5.15, 1.5 * n), subplot_kw={'projection': ccrs.Mercator()})

    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row, t in enumerate(time_indices):
        orig_map = z_unstacked_orig.isel(time=t)
        recon_map = z_unstacked_recon.isel(time=t)

        date = pd.to_datetime(z_stack_original.time.values[t])
        date_label = date.strftime("%Y-%m")  # nur Jahr-Monat, Tag weg

        for col, (data, label) in enumerate(zip([orig_map, recon_map], ["Original", "Reconstructed"])):
            ax = axes[row, col]
            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False
            )
            ax.coastlines()
            ax.set_title(label if row == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            if col == 0:
                ax.text(
                    -0.05, 0.5, date_label,
                    transform=ax.transAxes,
                    fontsize=10,
                    va='center',
                    ha='right',
                    rotation=90
                )

    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Z-score")

    fig.subplots_adjust(wspace=0.02, hspace=0.05, left=0, right=1, top=0.96, bottom=0.12)
    plt.show()


def plot_average_loss_map_from_data(original_z, reconstructed_z, labels, cmin=None, cmax=None, ncols=3):
    """
    Plots average MSE loss maps per cluster.

    original_z: xarray.DataArray (time, location) – original standardized values
    reconstructed_z: xarray.DataArray (time, location) – predicted values
    labels: np.ndarray of shape (time,) – cluster labels
    cmin, cmax: optional values for color scale
    """

    orig = original_z.unstack("location").sortby(["latitude", "longitude"])
    recon = reconstructed_z.unstack("location").sortby(["latitude", "longitude"])

    # Loss.
    mse = (orig - recon) ** 2
    mse = mse.assign_coords(cluster=("time", labels))

    # Average map per cluster.
    cluster_maps = {}
    for cid in np.unique(labels):
        avg_map = mse.where(mse.cluster == cid).mean("time").sortby(["latitude", "longitude"])
        cluster_maps[cid] = avg_map

    sorted_ids = sorted(cluster_maps.keys())
    cluster_maps_ordered = [cluster_maps[i] for i in sorted_ids]

    vmin = cmin if cmin is not None else min([m.min().item() for m in cluster_maps_ordered])
    vmax = cmax if cmax is not None else max([m.max().item() for m in cluster_maps_ordered])

    # Layout
    n_clusters = len(cluster_maps_ordered)
    # ncols = 3
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.5 * nrows),
                             subplot_kw={'projection': ccrs.Mercator()})
    axes = axes.flatten()

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
        ax.text(
            0.03, 0.03, str(sorted_ids[i]),
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color='black',
            ha='left',
            va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.6)
        )

    for ax in axes[len(cluster_maps_ordered):]:
        ax.set_visible(False)

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="MSE")
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, top=0.98, bottom=0.12)
    plt.show()

def get_latents_and_cluster(model, X, k=9, batch_size=512, device="cpu"):
    """
    Computes the latent means (mu) for all inputs and clusters them using KMeans.
    """

    model.eval()
    model.to(device)

    if isinstance(X, torch.Tensor):
        X_tensor = X
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)

    X_tensor = X_tensor.to(device)
    latents = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            
            # Dein VAE Encoder-Forward
            h = model.encoder(batch)
            mu = model.fc_mean(h)
            latents.append(mu.cpu())

    latents = torch.cat(latents, dim=0)  # (n_samples, latent_dim)

    labels = apply_kmeans(latents, k)
    labels += 1

    return latents, labels

def plot_metrics(metric_list: list, title):
    """
    Plots Metrics of a models training.
    
    -metric_list: List of tupels. [([2,5,1,2], "Training Loss"),(...)]
    """

    plt.figure(figsize=(6, 3))
    
    for tupel in metric_list:
        plt.plot(tupel[0], label=tupel[1], linewidth=2)
        
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




#------------------ Hungarian Algorithm ------------------

def load_clusterseries(folder_path: str, reference: str):
    """
    Loads the "reference" cluster-series from the given folder-path.
    """

    for file_name in os.listdir(folder_path):
        if file_name == reference:
            full_path = os.path.join(folder_path, file_name)
            arr = np.load(full_path)
            return arr

def find_best_permutation(reference, target, n_clusters):
    """
    Finds the permutation, which aligns the target's series to the reference's series
    in the best way, using the Hungarian Algorithm.
    """
    # Costmatrix.
    cost_matrix = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = -np.sum((reference == i+1) & (target == j+1))

    # Hungarian Algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Permutation-Table.
    permutation = dict(zip(col_ind + 1, row_ind + 1))
    # Do the permutation.
    target_aligned = np.array([permutation[label] for label in target])
    
    return target_aligned, permutation