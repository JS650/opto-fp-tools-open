import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import pandas as pd
import numpy as np
from typing import List


# Helper function
def list_datasets(h5_file, group="/DataAcquisition", dataset_list=None) -> List[str]:
        """Recursively lists all datasets in the specified HDF5 group."""
        if dataset_list is None:
            dataset_list = []
        if group in h5_file:
            for key, item in h5_file[group].items():
                path = f"{group}/{key}".replace("//", "/")
                if isinstance(item, h5py.Dataset):
                    dataset_list.append(path)
                elif isinstance(item, h5py.Group):
                    list_datasets(h5_file, path, dataset_list)
        return dataset_list


def hdf5_to_parquet(standard_columns, input_file: str, output_file: str, conversion_file: str, chunk_size: int = 10000):    
    # Print status updates
    print('Running hdf5_to_parquet')
    print(f'Input file: {input_file}')
    print(f'Output file: {output_file}')
    print(f'Conversion file: {conversion_file}')
    print(' ')

    # Load conversion mapping (original -> new column names)
    conv_data = pd.read_csv(conversion_file)
    conv_map = dict(zip(conv_data['original'], conv_data['new']))

    with h5py.File(input_file, "r") as f:
        if 'DFF' in input_file:
            datasets = list_datasets(f, "/DataProcessed")
        else:
            datasets = list_datasets(f, "/DataAcquisition")
        if not datasets:
            raise ValueError("No datasets found.")

        max_len = max(f[ds].shape[0] for ds in datasets)

        # Build a list of output columns and map of dataset -> final column names
        dataset_shapes = []
        final_columns = []

        for ds in datasets:
            #print(ds)
            data = f[ds]
            # Check name of dataset. If it contains DFFSignals and a number directly following,
            # remove that number for mapping purposes
            if 'DFFSignals' in ds:
                #base_name = ds.split('DFFSignals')[0] + 'DFFSignals' + ds.split('DFFSignals')[1][1:] # NOTE: Assumes single digit
                # If multi-digit numbers used, use regex instead as follows:
                import re
                base_name = re.sub(r'DFFSignals\d+', 'DFFSignals', ds)
            else:
                base_name = ds
            if data.ndim == 1:
                target_name = conv_map.get(base_name)
                if target_name and target_name in standard_columns:
                    final_columns.append(target_name)
                    dataset_shapes.append((ds, [target_name]))
            elif data.ndim == 2:
                mapped_cols = [
                    conv_map.get(f"{base_name}_col{i}") 
                    for i in range(data.shape[1])
                ]
                # Filter out None or unmapped
                mapped_cols = [col for col in mapped_cols if col in standard_columns]
                if mapped_cols:
                    final_columns.extend(mapped_cols)
                    dataset_shapes.append((ds, mapped_cols))

        if not final_columns:
            raise ValueError("No matching columns found using the conversion file.")

        # Define the schema for Parquet
        schema = pa.schema([(col, pa.float64()) for col in standard_columns])
        writer = pq.ParquetWriter(output_file, schema=schema)

        for start in range(0, max_len, chunk_size):
            chunk_data = {col: np.full(chunk_size, np.nan) for col in standard_columns}

            for ds, mapped_cols in dataset_shapes:
                data = f[ds][start:start + chunk_size]
                if data.ndim == 1:
                    col = mapped_cols[0]
                    chunk_data[col][:data.shape[0]] = data
                elif data.ndim == 2:
                    for i, col in enumerate(mapped_cols):
                        if i < data.shape[1]:
                            chunk_data[col][:data.shape[0]] = data[:, i]

            # Convert to DataFrame and write
            chunk_df = pd.DataFrame(chunk_data)
            chunk_df = chunk_df.dropna(how="all")  # optional: skip completely empty chunks

            if not chunk_df.empty:
                table = pa.Table.from_pandas(chunk_df, schema=schema, preserve_index=False)
                writer.write_table(table)
                #print(f"Wrote rows {start}–{start + chunk_size} to Parquet.")

        writer.close()
    print(f"HDF5 streamed to Parquet with standardized columns and metadata: {output_file}")


# Example usage
if __name__ == "__main__":

    input_hdf5_file = '/Users/samlaxer/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/20251219_ScanDay/sub-46Ear435509_ses-20251219_run-6_desc-ModPowAndFreq_fp_0005_DFF2.doric'
    
    
    output_parquet_file = input_hdf5_file.replace('.doric', '.standardized.parquet')
    if 'DFF' in input_hdf5_file:
        conversion_csv_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/code/header_conversions_DFF.csv'
        STANDARD_COLUMNS = [
        'time_norm', '460-490nm_roi1_norm', '460-490nm_roi2_norm'
        ]
    else:
        conversion_csv_file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/code/header_conversions.csv'
        STANDARD_COLUMNS = [
            "400-410nm_time", "400-410nm_roi1", "400-410nm_roi2", "400-410nm_roi3",
            "460-490nm_time", "460-490nm_roi1", "460-490nm_roi2", "460-490nm_roi3",
            "555-570nm_time", "555-570nm_roi1", "555-570nm_roi2", "555-570nm_roi3",
            "ttl_time", "ttl_TR", "ttl_stim", "opto_DIO1", "opto_DIO2", "minicube_405nm", "minicube_465nm", "minicube_time",
            "time_norm", "460-490nm_roi1_norm", "460-490nm_roi2_norm", "460-490nm_roi3_norm",
            "555-570nm_roi1_norm", "555-570nm_roi2_norm", "555-570nm_roi3_norm",
            "minicube_465nm_norm"
        ]
    
    hdf5_to_parquet(
        standard_columns=STANDARD_COLUMNS,
        input_file=input_hdf5_file,
        output_file=output_parquet_file,
        conversion_file=conversion_csv_file,
        chunk_size=10000
    )

