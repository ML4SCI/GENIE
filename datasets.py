import os
import requests
import functools
import pathlib
import shutil
import logging
import awkward as ak
import pandas as pd
import numpy as np
import torch
# import tqdm.auto as tqdm
import dask as da
import h5py as hp
from preprocess_dask import _transform, _extract_coords
import pyarrow.parquet as pq
from pathlib import Path
import uproot
import dask.dataframe as dd
import dask_awkward
import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union
import gc

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
    Downloading the Dataset
'''

def download(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path

test_link = "https://zenodo.org/records/2603256/files/test.h5?download=1"
train_link = "https://zenodo.org/records/2603256/files/train.h5?download=1"
val_link = "https://zenodo.org/records/2603256/files/val.h5?download=1"

def process_chunk(chunk_data: pd.DataFrame, start: int, stop: int) -> dict:
    """Process a single chunk of data efficiently."""
    try:
        v = _extract_coords(chunk_data, start=start, stop=stop)
        return v
    except Exception as e:
        logging.error(f"Error processing chunk {start}-{stop}: {str(e)}")
        raise

def convert(source: Union[str, Path], 
           destdir: Union[str, Path], 
           basename: str, 
           start: Optional[int] = None, 
           stop: Optional[int] = None, 
           step: Optional[int] = None, 
           limit: Optional[int] = None,
           num_workers: int = 4) -> None:
    """
    Converts the DataFrame into an Awkward array and performs the read-write
    operations for the same. Also performs Batching of the file into smaller
    Awkward files.

    Args:
        source: Path to source H5 file
        destdir: Directory to save parquet files
        basename: Base name for output files
        start: Starting index for processing
        stop: Ending index for processing
        step: Size of chunks for parallel processing
        limit: Maximum number of events to process
        num_workers: Number of worker processes for parallel processing
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create destination directory
        destdir = Path(destdir)
        destdir.mkdir(parents=True, exist_ok=True)
        
        # Read H5 file using dask for lazy loading
        ddf = dd.read_hdf(source, key='table', start=start, stop=stop)
        if limit is not None:
            ddf = ddf.head(limit)
        
        total_events = len(ddf)
        logging.info(f'Total events: {total_events}')
        
        if step is None:
            step = min(1000, total_events)  # Default chunk size
            
        # Process chunks in parallel
        chunks = ddf.partitions(step)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                output_path = destdir / f"{basename}_{i}.parquet"
                future = executor.submit(process_chunk, chunk, i*step, (i+1)*step)
                futures.append((future, output_path))
            
            # Save results as they complete
            for future, output_path in tqdm.tqdm(futures, desc="Processing chunks"):
                try:
                    features = future.result()
                    ak.to_parquet(features, output_path, compression='snappy')
                    logging.info(f"Created parquet file: {output_path}")
                    # Force garbage collection after each chunk
                    gc.collect()
                except Exception as e:
                    logging.error(f"Error saving chunk to {output_path}: {str(e)}")
                    raise
                    
        logging.info("Conversion completed successfully")
        
    except Exception as e:
        logging.error(f"Error in conversion process: {str(e)}")
        raise

def parquet_handler(source_loc, dest_loc = None):
    parquet_dir = Path(source_loc)
    directory = source_loc.split('/')[-1]
    if not os.path.exists(dest_loc):
        os.makedirs(dest_loc)

    csv_path = str(os.path.join(dest_loc, '%s_processed.csv' % directory))
    if os.path.exists(csv_path):
        logging.info("... CSV file already exists, moving on...")
        pass
    else:
        for i, parquet_path in enumerate(parquet_dir.glob('%s_file_*.parquet' % directory)):
            df = pq.read_table(parquet_path).to_pandas()
            write_header = i == 0 # Write header only on the 0th file
            write_mode = 'w' if i == 0 else 'a' # 'write' mode for 0th file, 'append' for others
            df.to_csv(csv_path, mode=write_mode, header=write_header)

    return

if __name__ == "__main__":

    CURRENT_DIR = os.getcwd()
    print(CURRENT_DIR)
    # Define the new folder name and its path
    new_folder_name = "downloads"
    new_folder_path = os.path.join(CURRENT_DIR, new_folder_name)

    # Check if the folder exists, if not, create it
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True)

    # Set the PROJECT_DIR to the new folder path
    PROJECT_DIR = new_folder_path
    PARQUET_FILE_LOC = os.path.join(PROJECT_DIR, 'converted')
    download(test_link, os.path.join(PROJECT_DIR, 'test.h5'))
    download(train_link, os.path.join(PROJECT_DIR, 'train.h5'))
    download(val_link, os.path.join(PROJECT_DIR, 'val.h5'))

    # Call the function
    convert(
        source=os.path.join(PROJECT_DIR, 'train.h5'),
        destdir=os.path.join(PROJECT_DIR, 'converted', 'train'),
        basename='train_file',
        start=0,
        stop=100000,
        step=1000,
        limit=None,
        num_workers=4
    )
    
    convert(
        source=os.path.join(PROJECT_DIR, 'test.h5'),
        destdir=os.path.join(PROJECT_DIR, 'converted', 'test'),
        basename='test_file',
        start=0,
        stop=30000,
        step=100,
        limit=None,
        num_workers=4
    )
    
    convert(
        source=os.path.join(PROJECT_DIR, 'val.h5'),
        destdir=os.path.join(PROJECT_DIR, 'converted', 'val'),
        basename='val_file',
        start=0,
        stop=10000,
        step=1000,
        limit=None,
        num_workers=4
    )

""" DO NOT UNCOMMENT THE FOLLOWING SECTION TO RUN THE CODE FOR DEMONSTRATION. THIS SECTION IS FOR FUTURE DEVELOPMENT AND OPTIMIZATION PURPOSES 
    AND IS NOT USEFUL FOR THE CURRENT SCOPE OF THE OBJECTIVE."""
    # print(df['x'][1])
    # parquet_handler(source_loc = os.path.join(PROJECT_DIR, 'converted', 'train'), dest_loc = os.path.join(PROJECT_DIR, 'processed', 'train'))
    # parquet_handler(source_loc = os.path.join(PROJECT_DIR, 'converted', 'test'), dest_loc = os.path.join(PROJECT_DIR, 'processed', 'test'))
    # parquet_handler(source_loc = os.path.join(PROJECT_DIR, 'converted', 'val'), dest_loc = os.path.join(PROJECT_DIR, 'processed', 'val'))

    # root_handler(os.path.join(PROJECT_DIR, 'prep'), os.path.join(PROJECT_DIR, 'prep'))