"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import contextlib
import hashlib
import os
import pickle
import shutil
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms
from tqdm import tqdm


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class DATASET_FLAGS(NamedTuple):
    DATA_SEED = 42
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
    CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)


class CustomCIFAR10(tv_datasets.CIFAR10):
    def __init__(self, use_separate_transform=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_transform = None
        if use_separate_transform:
            self.base_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
                    ),
                ]
            )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, base_image) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets that return a PIL Image
        img = Image.fromarray(img)
        base_img = img

        if self.transform is not None:
            img = self.transform(img)

        if self.base_transform is not None:
            base_img = self.base_transform(base_img)
        else:
            base_img = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img


class CustomCIFAR100(tv_datasets.CIFAR100):
    def __init__(self, use_separate_transform=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_transform = None
        if use_separate_transform:
            self.base_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        DATASET_FLAGS.CIFAR100_MEAN, DATASET_FLAGS.CIFAR100_STD
                    ),
                ]
            )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, base_image) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets that return a PIL Image
        img = Image.fromarray(img)
        base_img = img

        if self.transform is not None:
            img = self.transform(img)

        if self.base_transform is not None:
            base_img = self.base_transform(base_img)
        else:
            base_img = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img


class PairedCustomCIFAR100(tv_datasets.CIFAR100):
    def __init__(self, size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if size == -1:
            size = 32

        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(size),
            ]
        )
        if size != 32:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                ]
            )
        else:
            self.base_resize_transform = None

        self.base_finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    DATASET_FLAGS.CIFAR100_MEAN, DATASET_FLAGS.CIFAR100_STD
                ),
            ]
        )
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean if mean is not None else DATASET_FLAGS.CIFAR100_MEAN,
                    std if std is not None else DATASET_FLAGS.CIFAR100_STD,
                ),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, base_image) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:  # initial augmentation, no resizing
            img = self.transform(img)

        if self.base_resize_transform is not None:
            base_img = self.base_finishing_transform(self.base_resize_transform(img))
            img = self.finishing_transform(self.resize_transform(img))
        else:
            img = self.finishing_transform(self.resize_transform(img))
            base_img = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img


class PairedCustomCIFAR10(tv_datasets.CIFAR10):
    def __init__(self, size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if size == -1:
            size = 32

        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(size),
            ]
        )
        if size != 32:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                ]
            )
        else:
            self.base_resize_transform = None

        self.base_finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
                ),
            ]
        )
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean if mean is not None else DATASET_FLAGS.CIFAR10_MEAN,
                    std if std is not None else DATASET_FLAGS.CIFAR10_STD,
                ),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, base_image) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:  # initial augmentation, no resizing
            img = self.transform(img)

        if self.base_resize_transform is not None:
            base_img = self.base_finishing_transform(self.base_resize_transform(img))
            img = self.finishing_transform(self.resize_transform(img))
        else:
            img = self.finishing_transform(self.resize_transform(img))
            base_img = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img


class PairedImageFolder(tv_datasets.ImageFolder):
    def __init__(self, size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if size == -1:
            size = 32

        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(size),
            ]
        )
        if size != 32:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                ]
            )
        else:
            self.base_resize_transform = None

        self.base_finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
                ),
            ]
        )
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean if mean is not None else DATASET_FLAGS.CIFAR10_MEAN,
                    std if std is not None else DATASET_FLAGS.CIFAR10_STD,
                ),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, base_sample) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:  # initial augmentation, no resizing
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.base_resize_transform is not None:
            base_img = self.base_finishing_transform(self.base_resize_transform(img))
            img = self.finishing_transform(self.resize_transform(img))
        else:
            img = self.finishing_transform(self.resize_transform(img))
            base_img = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img


def get_cinic10(locator="cinic10/0_16", size=-1, data_root="./data"):
    dataset_name = locator.split("/")[0]
    pkeep = 0.5
    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))

    if size == -1:
        size = 32
    mean, std = DATASET_FLAGS.CINIC10_MEAN, DATASET_FLAGS.CINIC10_STD

    # Get train/test transforms, #resizing is done by the dataset
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    )
    transform_test = None  # transforms.Compose( [transforms.CenterCrop(32),])
    transform_vanilla = None  # transforms.Compose([transforms.CenterCrop(32),])

    root_dir = os.path.join(data_root, "CINIC-10")

    private_public_dataset = PairedImageFolder(
        size=size,
        mean=mean,
        std=std,
        root=os.path.join(root_dir, "trainval"),
        transform=transform_train,
    )

    test_dataset = PairedImageFolder(
        size=size,
        mean=mean,
        std=std,
        root=os.path.join(root_dir, "test"),
        transform=transform_test,
    )

    master_keep_path = os.path.join(
        data_root, dataset_name, "{:d}".format(num_experiment), "master_keep.npy"
    )
    if os.path.exists(master_keep_path):
        # print('reloading master keep list')
        master_keep = np.load(master_keep_path)
    else:
        # print('remaking master keep list')
        os.makedirs(os.path.dirname(master_keep_path), exist_ok=True)
        # get private/public split for experiment
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            master_keep = np.random.uniform(
                size=(num_experiment, len(private_public_dataset)), low=0, high=1
            )
        order = master_keep.argsort(0)
        master_keep = order < int(pkeep * num_experiment)
        np.save(master_keep_path, master_keep)

    if int(experiment_idx) == int(num_experiment):
        print("SPECIAL-CASING THIS IS THE FULL EVALUATION/TRAINING DATASET")
        private_indices = list(np.arange(start=0, stop=32))
        public_indices = list(np.arange(start=0, stop=len(private_public_dataset)))
    else:
        keep = np.array(master_keep[experiment_idx], dtype=bool)
        private_indices = list(np.where(keep)[0])
        public_indices = list(np.where(~keep)[0])

    public_dataset = torch.utils.data.Subset(private_public_dataset, public_indices)
    private_dataset = torch.utils.data.Subset(private_public_dataset, private_indices)

    transform_dict = {
        "train": transform_train,
        "test": transform_test,
        "vanilla": transform_vanilla,
    }

    num_classes = 10
    # return private_dataset, public_dataset, test_dataset, transform_dict, sample_input, num_classes
    return private_dataset, public_dataset, test_dataset, transform_dict, num_classes


def get_data(
    split_frac=0.5,
    dataset="cifar10/0_16",
    size=-1,
    data_root="./data",
    use_augmentation=True,
):
    if dataset.startswith("cifar"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
            num_classes,
        ) = get_cifar(locator=dataset, size=size, data_root=data_root)
    elif dataset.startswith("imagenet"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
            num_classes,
        ) = get_imagenet(
            locator=dataset,
            size=size,
            data_root=data_root,
            use_augmentation=use_augmentation,
        )
    elif dataset.startswith("cinic10"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
            num_classes,
        ) = get_cinic10(locator=dataset, size=size, data_root=data_root)
    full_dataset = private_dataset
    if public_dataset is None:
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            indices = np.random.permutation(len(private_dataset))

        th_indices = int(len(indices) * split_frac)
        indices_a = indices[:th_indices]
        indices_b = indices[th_indices:]

        public_dataset = torch.utils.data.Subset(private_dataset, indices_b)
        private_dataset = torch.utils.data.Subset(private_dataset, indices_a)

    dataset_dict = {
        "private": private_dataset,
        "public": public_dataset,
        "test": test_dataset,
        "full": full_dataset,
    }
    return dataset_dict, transform_dict, num_classes


def get_cifar(locator="cifar10/0_16", size=-1, data_root="./data"):
    if locator.split("/")[0] == "cifar10":
        dataset_name = "cifar10"
        dataset_fn = PairedCustomCIFAR10
        mean, std = DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
        if size == -1:
            size = 32
            mean, std = DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
        else:
            mean, std = DATASET_FLAGS.IMAGENET_MEAN, DATASET_FLAGS.IMAGENET_STD
        num_classes = 10
    else:
        dataset_name = "cifar100"
        # dataset_fn = tv_datasets.CIFAR100
        dataset_fn = PairedCustomCIFAR100
        if size == -1:
            size = 32
            mean, std = DATASET_FLAGS.CIFAR100_MEAN, DATASET_FLAGS.CIFAR100_STD
        else:
            mean, std = DATASET_FLAGS.IMAGENET_MEAN, DATASET_FLAGS.IMAGENET_STD
        num_classes = 100

    pkeep = 0.5
    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))

    # Get train/test transforms, #resizing is done by the dataset
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    )
    transform_test = None  # transforms.Compose( [transforms.CenterCrop(32),])
    transform_vanilla = None  # transforms.Compose([transforms.CenterCrop(32),])

    private_public_dataset = dataset_fn(
        size=size,
        mean=mean,
        std=std,
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = dataset_fn(
        size=size,
        mean=mean,
        std=std,
        root=data_root,
        train=False,
        transform=transform_test,
    )

    transform_dict = {"train": transform_train, "test": transform_test}

    master_keep_path = os.path.join(
        data_root, dataset_name, "{:d}".format(num_experiment), "master_keep.npy"
    )
    if os.path.exists(master_keep_path):
        # print('reloading master keep list')
        master_keep = np.load(master_keep_path)
    else:
        # print('remaking master keep list')
        os.makedirs(os.path.dirname(master_keep_path), exist_ok=True)
        # get private/public split for experiment
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            master_keep = np.random.uniform(
                size=(num_experiment, len(private_public_dataset)), low=0, high=1
            )

        order = master_keep.argsort(0)
        master_keep = order < int(pkeep * num_experiment)
        np.save(master_keep_path, master_keep)

    if int(experiment_idx) == int(num_experiment):
        print("SPECIAL-CASING THIS IS THE FULL EVALUATION/TRAINING DATASET")
        private_indices = list(np.arange(start=0, stop=32))
        public_indices = list(np.arange(start=0, stop=len(private_public_dataset)))
    else:
        keep = np.array(master_keep[experiment_idx], dtype=bool)
        private_indices = list(np.where(keep)[0])
        public_indices = list(np.where(~keep)[0])

    public_dataset = torch.utils.data.Subset(private_public_dataset, public_indices)
    private_dataset = torch.utils.data.Subset(private_public_dataset, private_indices)

    return private_dataset, public_dataset, test_dataset, transform_dict, num_classes


def get_imagenet(
    locator="imagenet-1k/0_16", size=-1, data_root="./data", use_augmentation=True
):
    mean = DATASET_FLAGS.IMAGENET_MEAN
    std = DATASET_FLAGS.IMAGENET_STD
    num_classes = 1000

    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))
    dataset_name = locator.split("/")[0]

    if size == -1:
        size = 224
    resize_size = int(256.0 / 224 * size)

    # Get train/test transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if not use_augmentation:
        transform_train = transform_test
    transform_dict = {
        "train": transform_train,
        "test": transform_test,
        "vanilla": transform_test,
    }

    root_dir = os.path.join(data_root, dataset_name)
    private_public_dataset = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, "train"), transform=transform_train
    )
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, "test"), transform=transform_test
    )

    # get private/public split for experiment
    pkeep = 0.5

    master_keep_path = os.path.join(
        data_root, dataset_name, "{:d}".format(num_experiment), "master_keep.npy"
    )
    if os.path.exists(master_keep_path):
        master_keep = np.load(master_keep_path)
    else:
        os.makedirs(os.path.dirname(master_keep_path), exist_ok=True)
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            master_keep = np.random.uniform(
                size=(num_experiment, len(private_public_dataset)), low=0, high=1
            )
        order = master_keep.argsort(0)
        master_keep = order < int(pkeep * num_experiment)
        np.save(master_keep_path, master_keep)

    if int(experiment_idx) == int(num_experiment):
        print("SPECIAL-CASING THIS IS THE FULL EVALUATION/TRAINING DATASET")
        private_indices = list(np.arange(start=0, stop=32))
        public_indices = list(np.arange(start=0, stop=len(private_public_dataset)))

    else:
        keep = np.array(master_keep[experiment_idx], dtype=bool)
        private_indices = list(np.where(keep)[0])
        public_indices = list(np.where(~keep)[0])

    public_dataset = torch.utils.data.Subset(private_public_dataset, public_indices)
    private_dataset = torch.utils.data.Subset(private_public_dataset, private_indices)

    return private_dataset, public_dataset, test_dataset, transform_dict, num_classes


def huggingface_to_imagefolder_map(
    local_dataset_name, dataset_name, overwrite=True, splits=None, data_root="./data"
):
    root_dir = os.path.join(data_root, local_dataset_name)
    if splits is None:
        splits = [
            ("train", "train"),
            ("test", "validation"),  # actual test split has no labels
        ]
    str2int = None
    int2str = None

    def save_map(example_batch, dst_path, int2str, overwrite):
        """Apply save mapping across a sample."""
        final_path = os.path.join(
            dst_path,
            int2str[example_batch["label"]],
            "{}.{}".format(
                hashlib.md5(example_batch["image"].tobytes()).hexdigest(),
                example_batch["image"].format,
            ),
        )
        if overwrite or not os.path.exists(final_path):
            example_batch["image"].save(final_path)

        return example_batch

    if (not os.path.exists(root_dir)) or overwrite:
        print("STARTING to copy")
        try:
            shutil.rmtree(root_dir)
        except FileNotFoundError:
            pass
        for dst_split, hf_split in splits:  # loop through every split
            ds = load_dataset(dataset_name, split=hf_split)
            if str2int is None:  # Get label mappings
                str2int = {
                    k.strip().replace(" ", "_").replace("'", "").replace(",", "_"): v
                    for k, v in ds.features["label"]._str2int.items()
                }
                int2str = [
                    k.strip().replace(" ", "_").replace("'", "").replace(",", "_")
                    for k in ds.features["label"]._int2str
                ]
            for label in int2str:  # Create label folders
                os.makedirs(os.path.join(root_dir, dst_split, label), exist_ok=True)
            dst_path = os.path.join(root_dir, dst_split)
            ds.map(lambda x: save_map(x, dst_path, int2str, overwrite), num_proc=16)
        with open(os.path.join(root_dir, "class_idx.pkl"), "wb") as f:
            pickle.dump(str2int, f)
    return root_dir


from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


def check_Image(path):
    try:
        Image.open(path).verify()
    except:
        return False
    return True


def imagefolder_verification(root_dir, delete_broken=False):
    # run verification pass
    from glob import glob

    image_paths = glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
    image_paths = [ip for ip in image_paths if os.path.isfile(ip)]
    total_images = len(image_paths)
    invalid_images = 0
    invalid_image_paths = []
    pbar = tqdm(image_paths)
    for ip in pbar:
        is_valid = check_Image(ip)
        if not is_valid:
            invalid_images += 1
            invalid_image_paths.append(ip)
            if delete_broken:
                try:
                    os.remove(ip)
                except:
                    pass
        pbar.set_postfix_str("INVALID/TOTAL {}/{}".format(invalid_images, total_images))
    print("Done with verification")
    print(invalid_image_paths)


def get_imagefolder_dataset(dataset_name, size=-1, data_root="./data"):
    if size == -1:
        size = 224
        resize_size = 256
    else:
        resize_size = size

    if dataset_name == "cifar-10":
        mean = DATASET_FLAGS.CIFAR10_MEAN
        std = DATASET_FLAGS.CIFAR10_STD
        num_classes = 10
    elif dataset_name == "cifar-100":
        mean = DATASET_FLAGS.CIFAR100_MEAN
        std = DATASET_FLAGS.CIFAR100_STD
        num_classes = 100
    elif dataset_name == "imagenet-1k":
        mean = DATASET_FLAGS.IMAGENET_MEAN
        std = DATASET_FLAGS.IMAGENET_STD
        num_classes = 1000
    else:
        raise NotImplementedError

    transform_train = transforms.Compose(
        [
            transforms.Resize(resize_size),
            # transforms.CenterCrop(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # sample_input = torch.rand((1, 3, size, size))
    transform_dict = {"train": transform_train, "test": transform_test}

    root_dir = os.path.join(data_root, dataset_name)
    private_dataset = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, "train"), transform=transform_train
    )
    public_dataset = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, "val"), transform=transform_train
    )
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, "test"), transform=transform_test
    )

    dataset_dict = {
        "private": private_dataset,
        "public": public_dataset,
        "test": test_dataset,
    }
    # return dataset_dict, transform_dict, sample_input, num_classes
    return dataset_dict, transform_dict, num_classes


def set_transform(dataset, transform):
    if hasattr(dataset, "transform"):
        dataset.transform = transform
    elif hasattr(dataset, "dataset"):
        if hasattr(dataset.dataset, "transform"):
            dataset.dataset.transform = transform
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return dataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        mode="mia",
        batch_size: int = 16,
        num_workers: int = 16,
        image_size: int = -1,
        data_root: str = "./data",
        use_augmentation: bool = True,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.image_size = image_size
        self.data_root = data_root
        self.use_augmentation = use_augmentation

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_dict, transform_dict, num_base_classes = get_data(
            split_frac=0.5,
            dataset=self.dataset_name,
            size=self.image_size,
            data_root=self.data_root,
            use_augmentation=self.use_augmentation,
        )

        self.num_base_classes = num_base_classes

        if self.mode == "mia":
            self.train_dataset = dataset_dict[
                "public"
            ]  # large dataset of samples that were not used to train the original network
            self.test_dataset = dataset_dict["private"]  # This was used to train the nw
            self.val_dataset = dataset_dict[
                "test"
            ]  # test samples to test generalization of score model

        elif self.mode == "eval":
            self.train_dataset = dataset_dict[
                "public"
            ]  # large dataset of samples that were not used to train the original network
            self.test_dataset = dataset_dict["private"]  # This was used to train the nw
            self.val_dataset = dataset_dict[
                "test"
            ]  # test samples to test generalization of score model
            if "vanilla" in transform_dict:
                self.train_dataset = set_transform(
                    self.train_dataset, transform_dict["vanilla"]
                )
                self.test_dataset = set_transform(
                    self.test_dataset, transform_dict["vanilla"]
                )
                self.val_dataset = set_transform(
                    self.val_dataset, transform_dict["vanilla"]
                )

        else:
            self.train_dataset = dataset_dict["private"]
            self.test_dataset = dataset_dict["public"]
            self.val_dataset = dataset_dict["test"]

        return

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return [
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
        ]
