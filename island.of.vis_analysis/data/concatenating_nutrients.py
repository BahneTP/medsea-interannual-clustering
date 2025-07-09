import os
import xarray as xr

def combine(path):
    datasets = []
    folder_path = os.path.abspath(path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".nc"):
            file_path = os.path.join(folder_path, filename)
            try:
                ds = xr.open_dataset(file_path)

                # Rundung der depth-Koordinate
                if 'depth' in ds.coords:
                    ds = ds.assign_coords(depth=ds['depth'].round(2))
                elif 'depth' in ds.dims:
                    ds['depth'] = ds['depth'].round(2)
                else:
                    print(f"Warning: No depth coord in {filename}")

                print(ds.depth)  # kontrolliere
                
                datasets.append(ds)

            except Exception as e:
                print(f"Error with {filename}: {e}")

    return xr.merge(datasets, combine_attrs="override")


ds = combine("nutrients")
print(ds)
print(ds.po4.depth)

ds.to_netcdf("nutrients_combined.nc", format="NETCDF4")