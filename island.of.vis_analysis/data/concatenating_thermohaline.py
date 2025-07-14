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

                datasets.append(ds)

            except Exception as e:
                print(f"Error with {filename}: {e}")
    return xr.merge(datasets, combine_attrs="override")

# Now merging.
reanalysis = combine("thermohaline/reanalysis")
reanalysis = reanalysis.sel(time=slice(None, "2023-05-31"))
print(reanalysis.dims)

forecasting = combine("thermohaline/forecasting")
forecasting = forecasting.sel(time=slice("2023-06-01", None))
print(forecasting.dims)

combined = xr.concat([reanalysis, forecasting], dim="time", combine_attrs="override")
combined = combined.sel(time=slice("1998-01-01", "2025-04-30"))
combined = combined.where(combined['depth'] <= 95, drop=True)

print(combined.time)
print(combined.thetao.values.shape)


# Now we need to ensure that we use the right missing_value's and _FillValue's for the saving.
for var in combined.data_vars:
    v = combined[var]
    v.encoding['_FillValue'] = 1.0e+20
    if 'missing_value' in v.encoding:
        del v.encoding['missing_value']
    v.attrs['missing_value'] = 1.0e+20  # optional, CF-Doku

combined.to_netcdf("combined_thermohaline.nc", format="NETCDF4")