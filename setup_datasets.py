import os
import shutil

import global_vars as Global
"""
Find permanant data storage location. 
Download compressed datasets if not present.
Copy compressed data to temporary storage location (on compute note).
Unpack data on temporary storage location 
"""
#STORAGE_PATH = "./workspace/datasets/"
#CACHE_PATH=STORAGE_PATH
STORAGE_PATH = "~/projects/rpp-bengioy/caotians"
CACHE_PATH = os.path.join(os.environ.get("SLURM_TMPDIR"), "data")

if __name__ == "__main__":
    assert os.path.exists(STORAGE_PATH)

    for dataset in Global.all_dataset_classes:
        if 'name' in dataset.__dict__:
            set = dataset(root_path=os.path.join(STORAGE_PATH, dataset.name), download=True, extract=False)
    shutil.copytree(STORAGE_PATH, CACHE_PATH)
    for dataset in Global.all_dataset_classes:
        if 'name' in dataset.__dict__:
            set = dataset(root_path=os.path.join(CACHE_PATH, dataset.name), download=True, extract=True)