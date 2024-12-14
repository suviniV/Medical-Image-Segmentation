import os
import sys

def get_patient_dirs(path, dirs):
    """Get all patient directories from BraTS 2020 dataset structure"""
    for item in os.listdir(path):
        if item.startswith('BraTS20_Training_'):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                dirs.append(item_path)
    return dirs

def _crawl(path, dirs):
    files = []
    nii_gzs = []
    for it in os.listdir(path):
        it_path = os.path.join(path, it)
        if os.path.isfile(it_path):
            if it.endswith('.nii.gz'):
                nii_gzs.append(it_path)
            else:
                files.append(it_path)
        elif os.path.isdir(it_path):
            dirs = _crawl(it_path, dirs)
    if len(nii_gzs) > 0 or len(files) > 0:
        new_dir = {'path': path, 'files': files, 'nii_gzs': nii_gzs}
        dirs.append(new_dir)
    return dirs

def crawl(path, item, nii_gzs):
    if item.endswith('.nii.gz'):
        nii_gz_path = os.path.join(path, item)
        nii_gzs.append(nii_gz_path)
        return nii_gzs
    else:
        new_path = os.path.join(path, item)
        if not os.path.isdir(new_path):
            return nii_gzs
        for it in os.listdir(new_path):
            nii_gzs = crawl(new_path, it, nii_gzs)
        return nii_gzs

def find_patients(dir_name):
    pats = []
    pats = get_patient_dirs(dir_name, pats)

    for p in pats:
        print(p)
    print('Found %i patient directories.' % len(pats))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python find_mha_files.py <path_to_dataset>')
        sys.exit(1)
    find_patients(sys.argv[1])