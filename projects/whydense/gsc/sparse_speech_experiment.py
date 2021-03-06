# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

from nupic.research.frameworks.pytorch.audio_transforms import (
    AddBackgroundNoiseOnSTFT,
    AddNoise,
    ChangeAmplitude,
    ChangeSpeedAndPitchAudio,
    DeleteSTFT,
    FixAudioLength,
    FixSTFTDimension,
    StretchAudioOnSTFT,
    TimeshiftAudioOnSTFT,
    ToMelSpectrogram,
    ToMelSpectrogramFromSTFT,
    ToSTFT,
    ToTensor,
)
from nupic.research.frameworks.pytorch.model_utils import (
    add_sparse_cnn_layer,
    add_sparse_linear_layer,
    set_random_seed,
)
from nupic.research.frameworks.pytorch.models.resnet_models import resnet9
from nupic.research.frameworks.pytorch.speech_commands_dataset import (
    BackgroundNoiseDataset,
    PreprocessedSpeechDataset,
    SpeechCommandsDataset,
)
from nupic.torch.modules import Flatten, rezero_weights, update_boost_strength


def get_logger(name, verbose):
    """Configure Logger based on verbose level (0: ERROR, 1: INFO, 2: DEBUG)"""
    logger = logging.getLogger(name)
    if verbose == 0:
        logger.setLevel(logging.ERROR)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    return logger


class SparseSpeechExperiment(object):
    """This experiment tests the Google Speech Commands dataset, available
    here:

    http://download.tensorflow.org/data/speech_commands_v0.01.tar
    """

    def __init__(self, config):
        """Called once at the beginning of each experiment."""
        self.start_time = time.time()
        self.logger = get_logger(config["name"], config.get("verbose", 2))
        self.logger.debug("Config: %s", config)

        # Setup random seed
        seed = config["seed"]
        set_random_seed(seed)

        # Get our directories correct
        self.data_dir = os.path.join(config["data_dir"], "speech_commands")
        self.use_preprocessed_dataset = False

        # Configure Model
        self.model_type = config["model_type"]
        self.log_interval = config["log_interval"]
        self.batches_in_epoch = config["batches_in_epoch"]
        self.batch_size = config["batch_size"]
        self.background_noise_dir = config["background_noise_dir"]
        cnn_input_shape = config.get("cnn_input_shape", (1, 32, 32))
        linear_n = config["linear_n"]
        linear_percent_on = config["linear_percent_on"]
        cnn_out_channels = config["cnn_out_channels"]
        cnn_percent_on = config["cnn_percent_on"]
        boost_strength = config["boost_strength"]
        weight_sparsity = config["weight_sparsity"]
        cnn_weight_sparsity = config["cnn_weight_sparsity"]
        boost_strength_factor = config["boost_strength_factor"]
        k_inference_factor = config["k_inference_factor"]
        use_batch_norm = config["use_batch_norm"]
        dropout = config.get("dropout", 0.0)

        self.load_datasets()

        model = nn.Sequential()

        if self.model_type == "cnn":
            # Add CNN Layers
            input_shape = cnn_input_shape
            cnn_layers = len(cnn_out_channels)
            if cnn_layers > 0:
                for i in range(cnn_layers):
                    in_channels, height, width = input_shape
                    add_sparse_cnn_layer(
                        network=model,
                        suffix=i + 1,
                        in_channels=in_channels,
                        out_channels=cnn_out_channels[i],
                        use_batch_norm=use_batch_norm,
                        weight_sparsity=cnn_weight_sparsity,
                        percent_on=cnn_percent_on[i],
                        k_inference_factor=k_inference_factor,
                        boost_strength=boost_strength,
                        boost_strength_factor=boost_strength_factor,
                    )

                    # Feed this layer output into next layer input
                    in_channels = cnn_out_channels[i]

                    # Compute next layer input shape
                    wout = (width - 5) + 1
                    maxpool_width = wout // 2
                    input_shape = (in_channels, maxpool_width, maxpool_width)

            # Flatten CNN output before passing to linear layer
            model.add_module("flatten", Flatten())

            # Add Linear layers
            input_size = np.prod(input_shape)
            for i in range(len(linear_n)):
                add_sparse_linear_layer(
                    network=model,
                    suffix=i + 1,
                    input_size=input_size,
                    linear_n=linear_n[i],
                    dropout=dropout,
                    weight_sparsity=weight_sparsity,
                    percent_on=linear_percent_on[i],
                    k_inference_factor=k_inference_factor,
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                )
                input_size = linear_n[i]

            # Output layer
            model.add_module(
                "output", nn.Linear(input_size, len(self.train_loader.dataset.classes))
            )
            model.add_module("softmax", nn.LogSoftmax(dim=1))

        elif self.model_type == "resnet9":
            model = resnet9(
                num_classes=len(self.train_loader.dataset.classes), in_channels=1
            )
        else:
            raise RuntimeError("Unknown model type")

        self.use_cuda = torch.cuda.is_available()
        self.logger.debug("use_cuda %s", self.use_cuda)
        if self.use_cuda:
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        self.logger.debug("device %s", self.device)
        if torch.cuda.device_count() > 1:
            self.logger.debug("Using %s GPUs", torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

        self.model = model
        self.logger.debug("Model: %s", self.model)
        self.learning_rate = config["learning_rate"]
        self.optimizer = self.create_optimizer(config, self.model)
        self.lr_scheduler = self.create_learning_rate_scheduler(config, self.optimizer)

    def save(self, checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def restore(self, checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

    def create_learning_rate_scheduler(self, params, optimizer):
        """Creates the learning rate scheduler and attach the optimizer."""
        lr_scheduler = params.get("lr_scheduler", None)
        if lr_scheduler is None:
            return None

        if lr_scheduler == "StepLR":
            lr_scheduler_params = (
                "{'step_size': 1, 'gamma':" + str(params["learning_rate_factor"]) + "}"
            )

        else:
            lr_scheduler_params = params.get("lr_scheduler_params", None)
            if lr_scheduler_params is None:
                raise ValueError(
                    "Missing 'lr_scheduler_params' for {}".format(lr_scheduler)
                )

        # Get lr_scheduler class by name
        clazz = eval("torch.optim.lr_scheduler.{}".format(lr_scheduler))

        # Parse scheduler parameters from config
        lr_scheduler_params = eval(lr_scheduler_params)

        return clazz(optimizer, **lr_scheduler_params)

    def create_optimizer(self, params, model):
        """Create a new instance of the optimizer."""
        lr = params["learning_rate"]
        print("Creating optimizer with learning rate=", lr)
        if params["optimizer"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
            )
        elif params["optimizer"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise LookupError("Incorrect optimizer value")

        return optimizer

    def train(self, epoch):
        """Train one epoch of this model by iterating through mini batches.

        An epoch ends after one pass through the training set, or if the
        number of mini batches exceeds the parameter "batches_in_epoch".
        """
        self.logger.info("epoch: %s", epoch)

        t0 = time.time()
        self.pre_epoch()

        self.logger.info(
            "Learning rate: %s",
            self.learning_rate
            if self.lr_scheduler is None
            else self.lr_scheduler.get_lr(),
        )

        self.model.train()
        for batch_idx, (batch, target) in enumerate(self.train_loader):
            data = batch["input"]
            if self.model_type in ["resnet9", "cnn"]:
                data = torch.unsqueeze(data, 1)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx >= self.batches_in_epoch:
                break

        self.post_epoch()

        self.logger.info("training duration: %s", time.time() - t0)

    def pre_epoch(self):
        # Update dataset epoch when using pre-processed speech dataset
        if self.use_preprocessed_dataset:
            t2 = time.time()
            self.train_loader.dataset.next_seed()
            self.validation_loader.dataset.next_seed()
            self.test_loader.dataset.next_seed()
            self.bg_noise_loader.dataset.next_seed()
            self.logger.debug(
                "Dataset Load time = {0:.3f} secs, ".format(time.time() - t2)
            )

    def post_epoch(self):
        self.model.apply(update_boost_strength)
        self.model.apply(rezero_weights)
        self.lr_scheduler.step()

    def test(self, test_loader=None):
        """Test the model using the given loader and return test metrics."""
        if test_loader is None:
            test_loader = self.test_loader

        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch, target in test_loader:
                data = batch["input"]
                if self.model_type in ["resnet9", "cnn"]:
                    data = torch.unsqueeze(data, 1)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.sampler)
        test_error = 100.0 * correct / len(test_loader.sampler)

        entropy = self.entropy()
        ret = {
            "total_correct": correct,
            "mean_loss": test_loss,
            "mean_accuracy": test_error,
            "entropy": float(entropy),
        }

        return ret

    def entropy(self):
        """Returns the current entropy."""
        entropy = 0
        for module in self.model.modules():
            if module == self.model:
                continue
            if hasattr(module, "entropy"):
                entropy += module.entropy()

        return entropy

    def run_background_noise_test(self):
        """Runs background noise test."""
        if self.bg_noise_loader is not None:
            return self.test(self.bg_noise_loader)
        return None

    def validate(self):
        """Run validation."""
        if self.validation_loader:
            return self.test(self.validation_loader)
        return None

    def run_noise_tests(self):
        """
        Test the model with different noise values and return test metrics.
        """
        ret = {}
        test_data_dir = os.path.join(self.data_dir, "test")
        n_mels = 32

        # Test with noise
        for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            # Create noise dataset with noise transform
            noise_transform = transforms.Compose(
                [
                    FixAudioLength(),
                    AddNoise(noise),
                    ToSTFT(),
                    ToMelSpectrogramFromSTFT(n_mels=n_mels),
                    DeleteSTFT(),
                    ToTensor("mel_spectrogram", "input"),
                ]
            )

            noise_dataset = SpeechCommandsDataset(
                test_data_dir, noise_transform, silence_percentage=0
            )

            noise_loader = DataLoader(
                noise_dataset,
                batch_size=self.batch_size,
                sampler=None,
                shuffle=False,
                pin_memory=self.use_cuda,
            )

            ret[noise] = self.test(noise_loader)

        return ret

    def load_datasets(self):
        """The GSC dataset specifies specific files to be used as training,
        test, and validation.  We assume the data has already been processed
        according to those files into separate train, test, and valid
        directories.

        For our experiment we use a subset of the data (10 categories
        out of 30), just like the Kaggle competition.
        """
        n_mels = 32

        # Check if using pre-processed data or raw data
        self.use_preprocessed_dataset = PreprocessedSpeechDataset.is_valid(
            self.data_dir
        )
        if self.use_preprocessed_dataset:
            train_dataset = PreprocessedSpeechDataset(self.data_dir, subset="train")
            validation_dataset = PreprocessedSpeechDataset(
                self.data_dir, subset="valid", silence_percentage=0
            )
            test_dataset = PreprocessedSpeechDataset(
                self.data_dir, subset="test", silence_percentage=0
            )
            bg_noise_dataset = PreprocessedSpeechDataset(
                self.data_dir, subset="noise", silence_percentage=0
            )
        else:
            train_data_dir = os.path.join(self.data_dir, "train")
            test_data_dir = os.path.join(self.data_dir, "test")
            validation_data_dir = os.path.join(self.data_dir, "valid")
            background_noise_dir = os.path.join(
                self.data_dir, self.background_noise_dir
            )

            data_augmentation_transform = transforms.Compose(
                [
                    ChangeAmplitude(),
                    ChangeSpeedAndPitchAudio(),
                    FixAudioLength(),
                    ToSTFT(),
                    StretchAudioOnSTFT(),
                    TimeshiftAudioOnSTFT(),
                    FixSTFTDimension(),
                ]
            )

            feature_transform = transforms.Compose(
                [
                    ToMelSpectrogramFromSTFT(n_mels=n_mels),
                    DeleteSTFT(),
                    ToTensor("mel_spectrogram", "input"),
                ]
            )

            train_dataset = SpeechCommandsDataset(
                train_data_dir,
                transforms.Compose(
                    [
                        data_augmentation_transform,
                        # Uncomment to allow adding BG noise during training
                        # add_bg_noise,
                        feature_transform,
                    ]
                ),
            )

            test_feature_transform = transforms.Compose(
                [
                    FixAudioLength(),
                    ToMelSpectrogram(n_mels=n_mels),
                    ToTensor("mel_spectrogram", "input"),
                ]
            )

            validation_dataset = SpeechCommandsDataset(
                validation_data_dir, test_feature_transform, silence_percentage=0
            )

            test_dataset = SpeechCommandsDataset(
                test_data_dir, test_feature_transform, silence_percentage=0
            )

            bg_dataset = BackgroundNoiseDataset(
                background_noise_dir, transforms.Compose([FixAudioLength(), ToSTFT()])
            )

            bg_noise_transform = transforms.Compose(
                [
                    FixAudioLength(),
                    ToSTFT(),
                    AddBackgroundNoiseOnSTFT(bg_dataset),
                    ToMelSpectrogramFromSTFT(n_mels=n_mels),
                    DeleteSTFT(),
                    ToTensor("mel_spectrogram", "input"),
                ]
            )

            bg_noise_dataset = SpeechCommandsDataset(
                test_data_dir, bg_noise_transform, silence_percentage=0
            )

        weights = train_dataset.make_weights_for_balanced_classes()
        sampler = WeightedRandomSampler(weights, len(weights))

        # print("Number of training samples=",len(train_dataset))
        # print("Number of validation samples=",len(validation_dataset))
        # print("Number of test samples=",len(test_dataset))

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler
        )

        self.validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, sampler=None, shuffle=False
        )

        self.bg_noise_loader = DataLoader(
            bg_noise_dataset, batch_size=self.batch_size, sampler=None, shuffle=False
        )
