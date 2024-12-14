import os
import six
import pickle
import SimpleITK as sitk
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder
import h5py
from scipy.ndimage import zoom
from find_mha_files import get_patient_dirs
from random import shuffle
from skimage import color
import copy
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
#drive.mount('/content/drive')

# Define base paths
DRIVE_PATH = '/content/drive/MyDrive/RT-MyCopy/CNNbasedMedicalSegmentation'
DATASET_PATH = os.path.join(DRIVE_PATH, 'Dataset/MICCAI_BraTS2020_TrainingData')

def one_hot(labels, n_classes):
    """Convert labels to one-hot encoding"""
    labels = np.array(labels).reshape(-1, 1)
    encoder = OneHotEncoder(categories=[range(n_classes)], sparse_output=False)
    one_hot_labels = encoder.fit_transform(labels)
    return one_hot_labels

def vis_col_im(im, gt):
    indices_0 = np.where(gt == 0) # nothing
    indices_1 = np.where(gt == 1) # necrosis
    indices_2 = np.where(gt == 2) # edema
    indices_3 = np.where(gt == 3) # non-enhancing tumor
    indices_4 = np.where(gt == 4) # enhancing tumor
    
    im = np.asarray(im, dtype='float32')
    im = im*1./im.max()
    rgb_image = color.gray2rgb(im)
    m0 = [1., 1., 1.]
    m1 = [1., 0., 0.]
    m2 = [0.2, 1., 0.2]
    m3 = [1., 1., 0.2]
    m4 = [1., 0.6, 0.2]
    
    im = rgb_image.copy()
    im[indices_0[0], indices_0[1], :] *= m0
    im[indices_1[0], indices_1[1], :] *= m1
    im[indices_2[0], indices_2[1], :] *= m2
    im[indices_3[0], indices_3[1], :] *= m3
    im[indices_4[0], indices_4[1], :] *= m4
    
    plt.imshow(im)
    plt.show()
    plt.close()

def col_im(im, gt):
    im = np.asarray(im, dtype='float32')
    im = im*1./im.max()
    rgb_image = color.gray2rgb(im)
    im = rgb_image.copy()
    
    if gt is None:
        return im
        
    indices_0 = np.where(gt == 0) # nothing
    indices_1 = np.where(gt == 1) # necrosis
    indices_2 = np.where(gt == 2) # edema
    indices_3 = np.where(gt == 3) # non-enhancing tumor
    indices_4 = np.where(gt == 4) # enhancing tumor
    
    m0 = [1., 1., 1.]
    m1 = [1., 0., 0.] # red: necrosis
    m2 = [0.2, 1., 0.2] # green: edema
    m3 = [1., 1., 0.2] # yellow: non-enhancing tumor
    m4 = [1., 0.6, 0.2] # orange: enhancing tumor
    
    im[indices_0[0], indices_0[1], :] *= m0
    im[indices_1[0], indices_1[1], :] *= m1
    im[indices_2[0], indices_2[1], :] *= m2
    im[indices_3[0], indices_3[1], :] *= m3
    im[indices_4[0], indices_4[1], :] *= m4
    
    return im

def vis_ims(im0, gt0, im1, gt1, title0='Original', title1='Transformed'):
    im0 = col_im(im0, gt0)
    im1 = col_im(im1, gt1)
    
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    plt.imshow(im0)
    a.set_title(title0)
    a = fig.add_subplot(1,2,2)
    plt.imshow(im1)
    a.set_title(title1)
    
    plt.show()
    plt.close()
    
def get_im_as_ndarray(image, downsize=False):
    ims = [image['Flair'], image['T1'], image['T1c'], image['T2']]
    if downsize:
        ims = [zoom(x, 0.5, order=1) for x in ims]
    im = np.array(ims, dtype='int16')
    
    return im

def get_gt(gt, n_classes, downsize=False):
    if not downsize:
        return gt
    original_shape = gt.shape
    gt_onehot = np.reshape(gt, (-1,))
    gt_onehot = np.reshape(one_hot(gt_onehot, n_classes), original_shape + (n_classes,))
    gt_onehot = np.transpose(gt_onehot, (3, 0, 1, 2))
    
    zoom_gt = np.array([zoom(class_map, 0.5, order=1) for class_map in gt_onehot])
    zoom_gt = zoom_gt.argmax(axis=0)
    zoom_gt = np.asarray(zoom_gt, dtype='int8')
    
    return zoom_gt

def convert_gt_to_onehot(gt, n_classes):
    """Convert ground truth to one-hot encoding"""
    if gt is None:
        return None
        
    # Check if gt is 3D (depth, height, width)
    if len(gt.shape) != 3:
        raise ValueError(f"Expected 3D ground truth array, got shape {gt.shape}")
    
    # Reshape to (depth*height*width,)
    gt_flat = gt.reshape(-1)
    
    # Convert to one-hot
    gt_onehot = one_hot(gt_flat, n_classes)
    
    return gt_onehot
    
def process_gt(gt, n_classes, downsize=False):
    if downsize:
        gt = zoom(gt, 0.5, order=0)
        gt = np.asarray(gt, dtype='int8')
    gt = np.transpose(gt, (1, 2, 0))
    l = np.reshape(gt, (-1,))
    l = np.reshape(one_hot(l, n_classes), (-1, n_classes))
    return l
    
def center(im):
    """Find the center coordinates of non-zero values in the image"""
    if im is None:
        raise ValueError("Input image is None")
    
    indices = np.where(im > 0)
    if len(indices[0]) == 0:
        raise ValueError("No non-zero values found in image")
        
    indices = np.array(indices)
    indices = indices.T
    return [int(i) for i in np.round(np.mean(indices, axis=0))]

def get_pats(dir_name):
    pats = []
    pats = get_patient_dirs(dir_name, pats)
    return pats

def get_im(pat_dir):
    """Get image data from BraTS 2020 dataset structure"""
    im = {'Flair': None, 'T1': None, 'T1c': None, 'T2': None, 'gt': None}
    
    # BraTS 2020 naming convention
    base_name = os.path.basename(pat_dir)
    flair_path = os.path.join(pat_dir, f"{base_name}_flair.nii")
    t1_path = os.path.join(pat_dir, f"{base_name}_t1.nii")
    t1ce_path = os.path.join(pat_dir, f"{base_name}_t1ce.nii")
    t2_path = os.path.join(pat_dir, f"{base_name}_t2.nii")
    seg_path = os.path.join(pat_dir, f"{base_name}_seg.nii")
    
    # Also check for .nii.gz files as fallback
    if not os.path.exists(flair_path):
        flair_path = os.path.join(pat_dir, f"{base_name}_flair.nii.gz")
    if not os.path.exists(t1_path):
        t1_path = os.path.join(pat_dir, f"{base_name}_t1.nii.gz")
    if not os.path.exists(t1ce_path):
        t1ce_path = os.path.join(pat_dir, f"{base_name}_t1ce.nii.gz")
    if not os.path.exists(t2_path):
        t2_path = os.path.join(pat_dir, f"{base_name}_t2.nii.gz")
    if not os.path.exists(seg_path):
        seg_path = os.path.join(pat_dir, f"{base_name}_seg.nii.gz")
    
    # Print paths for debugging
    print(f"Looking for files in {pat_dir}:")
    print(f"FLAIR: {os.path.exists(flair_path)}")
    print(f"T1: {os.path.exists(t1_path)}")
    print(f"T1ce: {os.path.exists(t1ce_path)}")
    print(f"T2: {os.path.exists(t2_path)}")
    print(f"Seg: {os.path.exists(seg_path)}")
    
    required_files = [flair_path, t1_path, t1ce_path, t2_path]
    if not all(os.path.exists(f) for f in required_files):
        print(f"Missing required files in {pat_dir}")
        return None
    
    try:
        im['Flair'] = sitk.GetArrayFromImage(sitk.ReadImage(flair_path))
        im['T1'] = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
        im['T1c'] = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_path))
        im['T2'] = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
        if os.path.exists(seg_path):
            im['gt'] = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        else:
            warnings.warn('Could not find ground truth. Is this a test image?')
    except Exception as e:
        print(f"Error reading files from {pat_dir}: {str(e)}")
        return None
    
    return im

def check(coords, shape):
    z, y, x = coords
    sl_z = (z-64, z+64)
    sl_y = (y-84, y+76) # -70, +90
    sl_x = (x-72, x+72)
    if sl_z[0] < 0:
        sl_z = (0, 128)
    elif sl_z[1] > shape[0]:
        sl_z = (shape[0]-128, shape[0])
        
    if sl_y[0] < 0:
        sl_y = (0, 160)
    elif sl_y[1] > shape[1]:
        sl_y = (shape[1]-160, shape[1])
        
    if sl_x[0] < 0:
        sl_x = (0, 144)
    elif sl_x[1] > shape[2]:
        sl_x = (shape[2]-144, shape[2])
        
    z_s = slice(sl_z[0], sl_z[1])
    y_s = slice(sl_y[0], sl_y[1])
    x_s = slice(sl_x[0], sl_x[1])
    
    return (z_s, y_s, x_s)
    
def create_folds(dir_name=DATASET_PATH):
    """Create folds for cross-validation using BraTS 2020 dataset"""
    pats = get_pats(dir_name)
    shuffle(pats)
    
    # Calculate fold sizes for 369 subjects
    fold_size = len(pats) // 3
    
    folds = []
    for i in range(3):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < 2 else len(pats)
        
        validation_slice = slice(start_idx, end_idx)
        valid_and_test = pats[validation_slice]
        mid = len(valid_and_test) // 2
        valid = valid_and_test[:mid]
        test = valid_and_test[mid:]
        train = pats[:start_idx] + pats[end_idx:]
        
        folds.append({
            'train': copy.deepcopy(train),
            'valid': copy.deepcopy(valid),
            'test': copy.deepcopy(test)
        })
    
    with open('folds.pkl', 'wb') as f:
        pickle.dump(folds, f)
    
    return folds

def get_image_slice(image):
    z_s, x_s, y_s = check(center(image['Flair']), image['Flair'].shape)
    im = {}
    for key, value in six.iteritems(image):
        if value is not None:
            im.update({key: value[z_s, x_s, y_s]})
    return im, (z_s, x_s, y_s)

def gen_images(dir_name=DATASET_PATH, n=1, specific=False, interval=None, crop=False, randomize=False, custom_pats=None):
    pats = get_pats(dir_name) if custom_pats is None else custom_pats
    print('%i images in total.' % len(pats))
    if randomize:
        print('shuffling patients.')
        shuffle(pats)
    
    if interval is None:
        a = 0
        b = n
    else:
        a, b = interval
    if b == -1:
        b = len(pats)
    if a == -1:
        pats = pats[::-1]
        a = 0
        print('yielding images in reverse order.')
    elif b > len(pats):
        raise ValueError('There are %i images but user requested %i.' % (len(pats), b))
        
    print('yielding images in range: (%i, %i).' % (a, b))
    for p in pats[a:b]:
        try:
            print('{}\t'.format(p))
            im = get_im(p)
            if im is None:
                print(f"Skipping {p} due to missing or invalid files")
                continue
                
            if not crop:
                yield im
            else:
                try:
                    z_s, x_s, y_s = check(center(im['Flair']), im['Flair'].shape)
                    for key in im:
                        if im[key] is not None:
                            im[key] = im[key][z_s, x_s, y_s]
                    yield im
                except ValueError as e:
                    print(f"Error processing {p}: {str(e)}")
                    continue
        except Exception as e:
            print('Problem with: ', p)
            print(f"Error: {str(e)}")
            continue

def get_shapes(im):
    shapes = []
    for key in im:
        shapes.append(im[key].shape)
    return shapes
    
def check_shapes(im):
    for sh in get_shapes(im):
        if sh != (128, 160, 144):
            return False
    return True
    
def test_shapes():
    count = 1
    errors = 0
    for image in gen_images(n=-1, crop=True):
        if not check_shapes(image):
            print('Problem with image %i.' % count)
            errors += 1
        else:
            print('image %i is ok.' % count)
        count += 1
    print('Finished with %i errors.' % errors)
    
def build_hdf5_from_fold(fold_number=0):
    """Create dataset from fold"""
    if not os.path.exists('folds.pkl'):
        print('Creating folds...')
        create_folds()
    
    with open('folds.pkl', 'rb') as f:
        folds = pickle.load(f)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(DRIVE_PATH, 'data/datasets')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'brats_fold{fold_number}.hdf5')
    data = h5py.File(output_path, 'w')
    
    depth, height, width = (128, 160, 144)  # Standard BraTS dimensions
    n_chans = 4  # 4 modalities: FLAIR, T1, T1ce, T2
    dimprod = height*width*depth
    n_classes = 5  # Background + 4 tumor subregions
    
    # Count valid images (with ground truth) first
    valid_counts = {'train': 0, 'valid': 0, 'test': 0}
    for key in ['train', 'valid', 'test']:
        for image in gen_images(custom_pats=folds[fold_number][key], crop=True, n=-1):
            if image['gt'] is not None:
                valid_counts[key] += 1
    
    # Create datasets with correct sizes
    x = data.create_dataset('train_x', (valid_counts['train'], depth, n_chans, height, width), dtype='int16')
    vx = data.create_dataset('valid_x', (valid_counts['valid'], depth, n_chans, height, width), dtype='int16')
    tx = data.create_dataset('test_x', (valid_counts['test'], depth, n_chans, height, width), dtype='int16')
    y = data.create_dataset('train_y', (valid_counts['train'], dimprod, n_classes), dtype='int8')
    vy = data.create_dataset('valid_y', (valid_counts['valid'], dimprod, n_classes), dtype='int8')
    ty = data.create_dataset('test_y', (valid_counts['test'], dimprod, n_classes), dtype='int8')

    dat_access = {
        'train': (x, y),
        'valid': (vx, vy),
        'test': (tx, ty)
    }
    
    for key in ['train', 'valid', 'test']:
        print('building %s set' % key)
        index = 0
        size = valid_counts[key]
        for image in gen_images(custom_pats=folds[fold_number][key], crop=True, n=-1):
            if image['gt'] is None:
                print('\tskipping image without ground truth')
                continue
                
            print('\treading image %i of %i...' % (index+1, size))
            gt = get_gt(image['gt'], n_classes, downsize=False)
            im = get_im_as_ndarray(image, downsize=False)
            
            dat_access[key][0][index, :, :, :, :] = np.transpose(im, (1, 0, 2, 3))
            dat_access[key][1][index, :, :] = convert_gt_to_onehot(gt, n_classes)
            index += 1        
    data.close()

def test_folds():
    folds = create_folds()
    
    for i, fold in enumerate(folds):
        print('Fold %i: ' % (i+1))
        for key in ['train', 'valid', 'test']:
            print('\t%s: ' % key)
            print('\t%i patients' % len(fold[key]))
            for patient in fold[key]:
                print('\t', patient)

if __name__ == '__main__':
    #test_shapes()
    #test_folds()
    #create_folds()
    build_hdf5_from_fold()