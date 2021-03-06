# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os

from ray import tune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
import networks


class Dataset:
    """Loads a dataset.

    Returns object with a pytorch train and test loader
    """

    def __init__(self, config=None):

        defaults = dict(
            dataset_name=None,
            data_dir=None,
            batch_size_train=128,
            batch_size_test=128,
            stats_mean=None,
            stats_std=None,
            augment_images=False,
        )
        defaults.update(config)
        self.__dict__.update(defaults)

        # expand ~
        self.data_dir = os.path.expanduser(self.data_dir)

        # recover mean and std to normalize dataset
        if not self.stats_mean or not self.stats_std:
            tempset = getattr(datasets, self.dataset_name)(
                root=self.data_dir, train=True, transform=transforms.ToTensor()
            )
            self.stats_mean = (tempset.data.float().mean().item() / 255,)
            self.stats_std = (tempset.data.float().std().item() / 255,)
            del tempset

        # set up transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.stats_mean, self.stats_std),
            ]
        )
        if not self.augment_images:
            aug_transform = transform
        else:
            aug_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.stats_mean, self.stats_std),
                ]
            )

        # load train set
        train_set = getattr(datasets, self.dataset_name)(
            root=self.data_dir, train=True, transform=aug_transform
        )
        self.train_loader = DataLoader(
            dataset=train_set, batch_size=self.batch_size_train, shuffle=True
        )

        # load test set
        test_set = getattr(datasets, self.dataset_name)(
            root=self.data_dir, train=False, transform=transform
        )
        self.test_loader = DataLoader(
            dataset=test_set, batch_size=self.batch_size_test, shuffle=False
        )


class Trainable(tune.Trainable):
    """ray.tune trainable generic class Adaptable to any pytorch module."""

    def __init__(self, config=None, logger_creator=None):
        tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)

    def _setup(self, config):
        network = getattr(networks, config["network"])(config=config)
        self.model = getattr(models, config["model"])(network, config=config)
        self.dataset = Dataset(config=config)
        self.model.setup()

    def _train(self):
        log = self.model.run_epoch(self.dataset)
        return log

    def _save(self, checkpoint_dir):
        self.model.save(checkpoint_dir)

    def _restore(self, checkpoint):
        self.model.restore(checkpoint)


def download_dataset(config):
    """Pre-downloads dataset.

    Required to avoid multiple simultaneous attempts to download same
    dataset
    """
    getattr(datasets, config["dataset_name"])(
        download=True, root=os.path.expanduser(config["data_dir"])
    )
