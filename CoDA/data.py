import os
import json
import torch
import numpy as np
import pandas as pd
import warnings
import torchvision.transforms as transforms
from misc import utils
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader

warnings.filterwarnings("ignore")
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'imagenet': [0.485, 0.456, 0.406]}
STDS = {'imagenet': [0.229, 0.224, 0.225]}


def _imagenet_class_file(spec):
    class_files = {
        'woof': 'class_woof.txt',
        'nette': 'class_nette.txt',
        'imagenet100': 'class100.txt',
        'imagenet1k': 'class_indices.txt',
        'IDC': 'class_IDC.txt',
        'imageA': 'imagenet-a.txt',
        'imageB': 'imagenet-b.txt',
        'imageC': 'imagenet-c.txt',
        'imageD': 'imagenet-d.txt',
        'imageE': 'imagenet-e.txt',
    }
    try:
        return os.path.join(os.path.dirname(__file__), 'misc', class_files[spec])
    except KeyError as error:
        raise AssertionError(f'spec does not exist!') from error


def _load_imagenet_class_ids():
    class_file = os.path.join(os.path.dirname(__file__), 'misc', 'class_indices.txt')
    with open(class_file, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


def _select_imagenet_classes(nclass, phase, seed, spec):
    all_classes = _load_imagenet_class_ids()
    if nclass >= len(all_classes):
        return all_classes

    phase = max(0, phase)
    cls_from = nclass * phase
    cls_to = nclass * (phase + 1)
    if seed == 0:
        with open(_imagenet_class_file(spec), 'r', encoding='utf-8') as file:
            classes = [line.strip() for line in file if line.strip()]
        classes = classes[cls_from:cls_to]
    else:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_classes))[cls_from:cls_to]
        classes = [all_classes[index] for index in indices]

    assert len(classes) == nclass
    return classes

class ImageFolder(datasets.DatasetFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None,
                 load_memory=False,
                 load_transform=None,
                 nclass=100,
                 phase=0,
                 slct_type='random',
                 seed=-1,
                 spec='none',
                 return_origin=False,
                 return_path=False,
                 mode_id_file=None):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        self.spec = spec
        self.return_origin = return_origin
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(nclass=nclass, phase=phase, seed=seed)
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.original_labels = self.find_original_classes()
        self.nclass = nclass
        cur_samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions, is_valid_file)
        self.samples = cur_samples
        self.targets = [s[1] for s in self.samples]
        self.original_targets = [self.original_labels[s[1]] for s in self.samples]

        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples
        self.return_path = return_path
        self.mode_id_file = mode_id_file
        if self.mode_id_file is not None:
            self.mode_id_df = pd.read_csv(self.mode_id_file)
            self.mode_id_df = self.mode_id_df.set_index("image_id")
            self.mode_ids = [self.mode_id_df.loc[s[0].split("/")[-1]]["mode_id"] for s in self.samples]

    def find_subclasses(self, nclass=100, phase=0, seed=0):
        """Finds the class folders in a dataset.
        """
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if seed == 0:
            if self.spec == 'woof':
                file_list = 'misc/class_woof.txt'
            elif self.spec == 'nette':
                file_list = 'misc/class_nette.txt'
            elif self.spec == 'imagenet100':
                file_list = 'misc/class100.txt'
            elif self.spec == 'imagenet1k':
                file_list = 'misc/class_indices.txt'
            elif self.spec == 'IDC':
                file_list = 'misc/class_IDC.txt'
            elif self.spec == 'imageA':
                file_list = 'misc/imagenet-a.txt'
            elif self.spec == 'imageB':
                file_list = 'misc/imagenet-b.txt'
            elif self.spec == 'imageC':
                file_list = 'misc/imagenet-c.txt'
            elif self.spec == 'imageD':
                file_list = 'misc/imagenet-d.txt'
            elif self.spec == 'imageE':
                file_list = 'misc/imagenet-e.txt'
            else:
                raise AssertionError(f'spec does not exist!')
            with open(file_list, 'r') as f:
                class_name = f.readlines()
            for c in class_name:
                c = c.split('\n')[0]
                classes.append(c)
            classes = classes[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass

        return classes, class_to_idx

    def find_original_classes(self):
        all_classes = sorted(os.listdir(self.root))
        original_labels = []
        for class_name in self.classes:
            original_labels.append(all_classes.index(class_name))
        return original_labels

    def _subset(self, slct_type='random', ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        # print("# examples in the smallest class: ", min_class)
        assert ipc <= min_class

        if slct_type == 'random':
            indices = np.arange(n)
        else:
            raise AssertionError(f'selection type does not exist!')

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory
        """
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            # if i % 100 == 0:
            #     print(f"Image loading.. {i}/{len(self.samples)}", end='\r')

        print(" " * 50, end='\r')
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
            image_id = path.split("/")[-1]
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        original_target = self.original_targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            original_target = self.target_transform(original_target)

        if self.mode_id_file is not None:
            if not self.load_memory:
                mode_id = self.mode_id_df.loc[image_id]['mode_id']
            else:
                mode_id = self.mode_ids[index]
            # Return original labels for DiT generation
            if self.return_origin:
                if self.return_path:
                    return sample, target, original_target, mode_id, path
                return sample, target, original_target, mode_id
            else:
                return sample, target, mode_id

        # Return original labels for DiT generation
        if self.return_origin:
            if self.return_path:
                return sample, target, original_target, path
            return sample, target, original_target
        else:
            return sample, target


class ImageNetJsonDataset(torch.utils.data.Dataset):
    """ImageNet stored in numbered shards with labels supplied by dataset.json."""

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 load_memory=False,
                 load_transform=None,
                 nclass=100,
                 phase=0,
                 slct_type='random',
                 seed=-1,
                 spec='none',
                 return_origin=False,
                 return_path=False,
                 mode_id_file=None):
        manifest_path = os.path.join(root, 'dataset.json')
        with open(manifest_path, 'r', encoding='utf-8') as file:
            manifest = json.load(file)
        labels = manifest.get('labels')
        if not isinstance(labels, list):
            raise ValueError(f"Invalid ImageNet manifest (missing labels list): {manifest_path}")
        if mode_id_file is not None:
            raise NotImplementedError('mode_id_file is not supported for dataset.json ImageNet sources.')

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_origin = return_origin
        self.return_path = return_path
        self.load_memory = load_memory
        self.load_transform = load_transform
        self.nclass = nclass
        self.spec = spec

        self.classes = _select_imagenet_classes(nclass, phase, seed, spec)
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}
        all_classes = _load_imagenet_class_ids()
        source_label_by_class = {class_name: index for index, class_name in enumerate(all_classes)}
        selected_source_labels = {
            source_label_by_class[class_name]: local_label
            for local_label, class_name in enumerate(self.classes)
        }
        self.original_labels = [source_label_by_class[class_name] for class_name in self.classes]

        self.samples = []
        for relative_path, source_label in labels:
            source_label = int(source_label)
            if source_label in selected_source_labels:
                self.samples.append((
                    os.path.join(root, *relative_path.split('/')),
                    selected_source_labels[source_label],
                ))
        if not self.samples:
            raise ValueError(
                f"No images for spec={spec}, nclass={nclass}, phase={phase} were found in {manifest_path}. "
                "Verify that dataset.json uses the standard ImageNet-1K label order."
            )

        self.targets = [target for _, target in self.samples]
        self.imgs = self._load_images(load_transform) if load_memory else self.samples

    def _load_images(self, transform=None):
        images = []
        for path, _ in self.samples:
            image = self.loader(path)
            if transform is not None:
                image = transform(image)
            images.append(image)
        print(" " * 50, end='\r')
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.load_memory:
            sample = self.imgs[index]
            path = self.samples[index][0]
        else:
            path, _ = self.samples[index]
            sample = self.loader(path)

        target = self.targets[index]
        original_target = self.original_labels[target]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            original_target = self.target_transform(original_target)

        if self.return_origin:
            if self.return_path:
                return sample, target, original_target, path
            return sample, target, original_target
        return sample, target


def create_imagenet_dataset(root, **kwargs):
    """Select the manifest-backed reader when an ImageNet dataset.json is present."""
    if os.path.isfile(os.path.join(root, 'dataset.json')):
        return ImageNetJsonDataset(root, **kwargs)
    return ImageFolder(root, **kwargs)


def transform_imagenet(size=-1,
                       augment=False,
                       from_tensor=False,
                       normalize=True,
                       rrc=True,
                       rrc_size=-1):

    resize_train = [transforms.Resize(size), transforms.CenterCrop(size)]
    resize_test = [transforms.Resize(size), transforms.CenterCrop(size)]

    if not augment:
        aug = []
    else:
        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[
                                      [-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203],
                                  ])
        aug = [transforms.RandomHorizontalFlip(), jittering, lighting]

        if rrc and size >= 0:
            if rrc_size == -1:
                rrc_size = size
            rrc_fn = transforms.RandomResizedCrop(rrc_size, scale=(0.5, 1.0))
            aug = [rrc_fn] + aug
        else:
            print("Dataset with basic imagenet augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(resize_train + cast + aug + normal_fn)
    test_transform = transforms.Compose(resize_test + cast + normal_fn)

    return train_transform, test_transform


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device('cpu'):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert != None:
                data = self.convert(data)
            yield data, target



def load_data(args, tsne=False,detailed=True):
    traindir = args.dataset_dir[0]
    valdir = args.dataset_dir[1]

    train_transform, test_transform = transform_imagenet(augment=args.augment,
                                                         size=args.size,
                                                         from_tensor=False)
    if args.nclass<=20 and args.size <= 256:
        args.load_memory = True
    train_dataset = create_imagenet_dataset(traindir,
                                            transform=train_transform,
                                            nclass=args.nclass,
                                            seed=args.dseed,
                                            slct_type=args.slct_type,
                                            load_memory=args.load_memory,
                                            spec=args.spec)
    val_dataset = create_imagenet_dataset(valdir,
                                          transform=test_transform,
                                          nclass=args.nclass,
                                          seed=args.dseed,
                                          load_memory=args.load_memory,
                                          spec=args.spec)

    nclass = len(train_dataset.classes)
    assert nclass == len(val_dataset.classes)
    for i in range(len(train_dataset.classes)):
        assert train_dataset.classes[i] == val_dataset.classes[i]
    assert np.array(train_dataset.targets).max() == nclass - 1
    assert np.array(val_dataset.targets).max() == nclass - 1

    if detailed:
        print("Subclass is extracted: ")
        print(" #class: ", nclass)
        print(" #train: ", len(train_dataset.targets))
    if args.ipc > 0 and detailed:
        print(f"  => subsample ({args.slct_type} ipc {args.ipc})")
    if detailed:
        print(" #valid: ", len(val_dataset.targets))

    # Use gradient accumulation at 1024 resolution.
    if args.size == 1024:
        args.batch_size = args.batch_size // args.accumulation_steps

    train_loader = MultiEpochsDataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         pin_memory=True)
    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size//2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=8,
                                       pin_memory=True)
    return train_dataset, train_loader, val_loader, args.nclass
