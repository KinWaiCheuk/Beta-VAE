import torch
import os
import numpy as np

class dSprites(torch.utils.data.Dataset):
    """`dSprites <https://github.com/deepmind/dsprites-dataset>` Dataset.

    Args:
        root (string): Root directory of dataset where ``dSprites/dsprites.pt`` exists.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, download=False):
        super(dSprites, self).__init__()
        self.root = root

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self.data = torch.load(os.path.join(self.raw_folder, 'dsprites.pt'))

    def __getitem__(self, index):
        img  = self.data[0][index]
        target = self.data[1][index]

        return img, target

    def __len__(self):
        return len(self.data[0]) # data is a tuple (img, target)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.raw_folder,
                                            'dsprites.pt'))
    def _check_npz_exists(self):
        return os.path.exists(os.path.join(self.raw_folder,
                                            'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))

    def download(self):
        """Download the dSprites data if it doesn't exist="""

        if self._check_exists():
            print('File already exists, skipping download')
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        data_path = os.path.join(self.raw_folder,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        
        if self._check_npz_exists():
            print('File already exists, skipping download')
            pass
        else:
            # download files
            print('Downloading...')
            url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
            torch.hub.download_url_to_file(url, data_path)

        
        print('Decompressing...')
        data = np.load(data_path)
        
        dataset = (torch.tensor(data['imgs']).float(),torch.tensor(data['latents_values']))
            

        with open(os.path.join(self.raw_folder, 'dsprites.pt'), 'wb') as f:
            torch.save(dataset, f) # saving is as .pt occupy a lot of space

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")