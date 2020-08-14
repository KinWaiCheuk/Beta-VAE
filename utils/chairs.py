import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity
import os
import numpy as np
from glob import glob
import PIL

class Chairs(VisionDataset):

    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
                # MD5 Hash                            Filename
                ("7941bff4d19665ed2dd8c82e88293e1b", "rendered_chairs.tar")]

    def __init__(self, root, transform=None, download=False):
        import pandas
        super(Chairs, self).__init__(root, transform=transform)

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.filename = glob(os.path.join(self.raw_folder, 'rendered_chairs', '**', '*.png'), recursive=True)

    def _check_integrity(self):
        for (md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.__class__.__name__, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return True

    def _check_processed(self):
        return os.path.isdir(os.path.join(self.raw_folder, "rendered_chairs"))   
    
    def download(self):
        import tarfile
        data_path = os.path.join(self.raw_folder,'rendered_chairs.tar')
        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            print('Downloading...')
            os.makedirs(self.raw_folder, exist_ok=True)
            url = "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"
            torch.hub.download_url_to_file(url, data_path)
            
        if self._check_processed():
            print('Files already processed')
        else:
            print(f'Processing...')    
            with tarfile.open(data_path) as f:
                f.extractall(os.path.join(self.raw_folder,'rendered_chairs'))
            print(f'Extraction done')

         
            
    def __getitem__(self, index):
        X = PIL.Image.open(self.filename[index])

        if self.transform is not None:
            X = self.transform(X)

        return X

    def __len__(self):
        return len(self.filename)
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__)