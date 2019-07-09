import os
import shutil

import global_vars as Global
"""
Unpack data on temporary storage location 
"""
CACHE_PATH = os.path.join(os.environ.get("SLURM_TMPDIR"),'data')

if __name__ == "__main__":
    assert os.path.exists(os.path.expanduser(CACHE_PATH))

    for dataset in Global.all_dataset_classes:
        if 'dataset_path' in dataset.__dict__:
            print("working on ", dataset.dataset_path)
            try:
                set = dataset(root_path=os.path.join(CACHE_PATH, dataset.dataset_path), download=False, extract=True)
            except RuntimeError:
                pass
            print("complete ", dataset.dataset_path)
