import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    try:
                        # Try multiple parsing strategies
                        seqtype = None
                        
                        # Strategy 1: Underscore-based parsing
                        if '_' in f:
                            parts = f.split('_')
                            if len(parts) > 3:
                                potential_type = parts[3].split('.')[0].lower()
                                if potential_type in ['t1', 't1ce', 't2', 'flair', 'seg']:
                                    seqtype = potential_type
                        
                        # Strategy 2: Filename-based parsing
                        if seqtype is None:
                            f_lower = f.lower()
                            if 't1ce' in f_lower:
                                seqtype = 't1ce'
                            elif 't1' in f_lower and 'ce' not in f_lower:
                                seqtype = 't1'
                            elif 't2' in f_lower:
                                seqtype = 't2'
                            elif 'flair' in f_lower:
                                seqtype = 'flair'
                            elif 'seg' in f_lower or 'mask' in f_lower:
                                seqtype = 'seg'
                        
                        # If we found a valid sequence type, add to datapoint
                        if seqtype:
                            datapoint[seqtype] = os.path.join(root, f)
                    except Exception as e:
                        print(f"Error processing file {f}: {e}")
                
                # Only add datapoint if it has all required sequence types
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)
                elif len(datapoint) > 0:
                    print(f"Incomplete datapoint: {datapoint.keys()}")

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    try:
                        # Try multiple parsing strategies
                        seqtype = None
                        
                        # Strategy 1: Underscore-based parsing
                        if '_' in f:
                            parts = f.split('_')
                            if len(parts) > 3:
                                potential_type = parts[3].split('.')[0].lower()
                                if potential_type in ['t1', 't1ce', 't2', 'flair', 'seg']:
                                    seqtype = potential_type
                        
                        # Strategy 2: Filename-based parsing
                        if seqtype is None:
                            f_lower = f.lower()
                            if 't1ce' in f_lower:
                                seqtype = 't1ce'
                            elif 't1' in f_lower and 'ce' not in f_lower:
                                seqtype = 't1'
                            elif 't2' in f_lower:
                                seqtype = 't2'
                            elif 'flair' in f_lower:
                                seqtype = 'flair'
                            elif 'seg' in f_lower or 'mask' in f_lower:
                                seqtype = 'seg'
                        
                        # If we found a valid sequence type, add to datapoint
                        if seqtype:
                            datapoint[seqtype] = os.path.join(root, f)
                    except Exception as e:
                        print(f"Error processing file {f}: {e}")
                
                # Only add datapoint if it has all required sequence types
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)
                elif len(datapoint) > 0:
                    print(f"Incomplete datapoint: {datapoint.keys()}")

    def __len__(self):
        return len(self.database) * 155

    def __getitem__(self, x):
        out = []
        n = x // 155
        slice = x % 155
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
