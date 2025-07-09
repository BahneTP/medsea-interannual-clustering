import os
import xarray as xr

#Data coordinates:
#N: 43,05341450446393 
#E: 16,09672139235972
#S: 42,944924514229555
#W: 15,987544756617531

def combine(path):
    datasets = []
    folder_path = os.path.abspath(path)

    for filename in os.listdir(folder_path):
        if filename.endswith(".nc"):
            file_path = os.path.join(folder_path, filename)
            try:
                ds = xr.open_dataset(file_path)

                # Delete bottomT if existing.
                if "bottomT" in ds:
                    ds = ds.drop_vars("bottomT")

                datasets.append(ds)

            except Exception as e:
                print(f"Error with {filename}: {e}")
    return xr.merge(datasets, combine_attrs="override")

# Now merging.
reanalysis = combine("thermohaline/reanalysis")
reanalysis = reanalysis.sel(time=slice(None, "2022-12-31"))
print(reanalysis.thetao.values.shape)

forecasting = combine("thermohaline/forecasting")
print(forecasting.thetao.values.shape)

combined = xr.concat([reanalysis, forecasting], dim="time", combine_attrs="override")
print(combined.thetao.values.shape)


# Now we need to ensure that we use the right missing_value's and _FillValue's for the saving.
for var in combined.data_vars:
    v = combined[var]
    v.encoding['_FillValue'] = 1.0e+20
    if 'missing_value' in v.encoding:
        del v.encoding['missing_value']
    v.attrs['missing_value'] = 1.0e+20  # optional, CF-Doku

combined.to_netcdf("combined.nc", format="NETCDF4")