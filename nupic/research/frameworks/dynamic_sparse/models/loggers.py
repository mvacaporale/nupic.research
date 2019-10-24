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

from collections import defaultdict

import numpy as np
import torch
from pandas import DataFrame


class BaseLogger:
    def __init__(self, model, config=None):
        defaults = dict(debug_weights=False, verbose=0)
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.model = model
        self.log = {}

    def log_pre_epoch(self):
        # reset log
        self.log = {}

    def log_post_epoch(self):
        if self.verbose > 0:
            print(self.log)
        if self.debug_weights:
            self.log_weights()

    def log_pre_batch(self):
        pass

    def log_post_batch(self):
        pass

    def log_metrics(self, loss, acc, train, noise):

        if train:
            self.log["train_loss"] = loss
            self.log["train_acc"] = acc
            if self.model.lr_scheduler:
                self.log["learning_rate"] = self.model.lr_scheduler.get_lr()[0]
            else:
                self.log["learning_rate"] = self.model.learning_rate
        else:
            if noise:
                self.log["noise_loss"] = loss
                self.log["noise_acc"] = acc
            else:
                self.log["val_loss"] = loss
                self.log["val_acc"] = acc

        if train and self.debug_weights:
            self.log_weights()

    def log_weights(self):
        """Log weights for all layers which have params."""
        if "param_layers" not in self.model.__dict__:
            self.model.param_layers = defaultdict(list)
            for m, ltype in [
                (m, self.model.has_params(m)) for m in self.model.network.modules()
            ]:
                if ltype:
                    self.model.param_layers[ltype].append(m)

        # log stats (mean and weight instead of standard distribution)
        for ltype, layers in self.model.param_layers.items():
            for idx, m in enumerate(layers):
                # keep track of mean and std of weights
                self.log[ltype + "_" + str(idx) + "_mean"] = torch.mean(m.weight).item()
                self.log[ltype + "_" + str(idx) + "_std"] = torch.std(m.weight).item()


class SparseLogger(BaseLogger):
    def __init__(self, model, config=None):
        print("SparseLogger.__init__")
        super().__init__(model, config)
        defaults = dict(
            log_magnitude_vs_coactivations=False,  # scatter plot of magn. vs coacts.
            debug_sparse=False,
            debug_network=False,
            log_sparse_layers_grid=False,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.model = model

        if self.debug_network:
            print(self.model.network)

        print(self.log_magnitude_vs_coactivations)
        if self.log_magnitude_vs_coactivations:
            for module in self.model.sparse_modules:
                print('     init coact tracking for ', module.__class__.__name__)
                module.init_coactivation_tracking()

    def log_metrics(self, loss, acc, train, noise):
        super().log_metrics(loss, acc, train, noise)
        if train and self.debug_sparse:
            self._log_sparse_levels()

        if train and self.log_magnitude_vs_coactivations:
            for module in self.model.sparse_modules:
                module.reset_coactivations()

        if not train and self._log_magnitude_and_coactivations:
            print('Logging mag and coactivations')
            self._log_magnitude_and_coactivations()

    def _log_magnitude_and_coactivations(self):
        print('_log_magnitude_and_coactivations')
        for module in self.model.sparse_modules:

            m = module.m
            if hasattr(m, "coactivations"):
                print('     m', m.__class__.__name__)
                # print('     c', m.coactivations[0:20])

            # coacts = m.coactivations.clone().detach().to("cpu").numpy()
            # weight = m.weight.clone().detach().to("cpu").numpy()
            # grads = m.weight.grad.clone().detach().to("cpu").numpy()

            # mask = ((m.weight.grad != 0)).to("cpu").numpy()

            # coacts = coacts[mask]
            # weight = weight[mask]
            # grads = grads[mask]
            # grads = np.log(np.abs(grads))

            # x, y, hue = "coactivations", "weight", "log_abs_grads"

            # dataframe = DataFrame(
            #     {x: coacts.flatten(), y: weight.flatten(), hue: grads.flatten()}
            # )
            # seaborn_config = dict(rc={"figure.figsize": (11.7, 8.27)}, style="white")

            # self.log["scatter_mag_vs_coacts_layer-{}".format(str(i))] = dict(
            #     data=dataframe, x=x, y=y, hue=hue, seaborn_config=seaborn_config
            # )
            # import ipdb; ipdb.set_trace()
            weight = m.weight.clone().detach().cpu()
            # grads = m.weight.grad.clone().detach().cpu()
            # mask = ((m.weight.grad != 0) & (m.weight != 0)).cpu()
            mask = (m.weight != 0).cpu()

            weight = weight[mask]
            # grads = grads[mask]

            # direction = torch.sign(weight) == torch.sign(grads)
            # direction[weight == 0] = grads[weight == 0] > 0
            # direction = direction.numpy()
            # direction = [
            #     "Growing" if d else "Shrinking"
            #     for d in direction
            # ]

            weight = weight.numpy()
            # grads = grads.numpy()
            # grads = np.log(np.abs(grads))

            # grads[grads < -10] = -10

            coact_d = {}
            coact_types = [
                "coactivations",
                "coacts_01",
                "coacts_10",
                "coacts_00",
            ]
            has_all = True
            for coact_type in coact_types:

                if not hasattr(m, coact_type):
                    has_all = False
                    continue

                coacts = getattr(m, coact_type)
                coacts = coacts.clone().detach().cpu().numpy()
                coacts = coacts[mask]

                coact_d[coact_type] = coacts

                # if "all" in coact_d:
                #     coact_d["all"] = np.vstack([
                #         coact_d["all"],
                #         coact_d[coact_type].flatten()
                #     ])
                # else:
                #     coact_d["all"] = coact_d[coact_type].flatten()

            log_name = "mag_vs_coacts_layer-{}".format(module.pos)
            self.log[log_name] = dict(
                coacts=DataFrame(coact_d),
                weights=weight,
            )

            # dominant = np.argmax(coact_d["all"], axis=0)
            # dmap = {
            #     0: "11_dominant",
            #     1: "01_dominant",
            #     2: "10_dominant",
            #     3: "00_dominant",
            # }
            # dominant = [
            #     dmap[d] for d in dominant
            # ]

            # for coact_type in coact_types:

            #     if coact_type not in coact_d:
            #         continue

            #     coacts = coact_d[coact_type]

            #     # x, y, hue = coact_type, "weight", "log_abs_grads"
            #     x, y = coact_type, "weight"

            #     dataframe = DataFrame({
            #         x: coacts.flatten(),
            #         y: weight.flatten(),
            #         # hue: grads.flatten()
            #     })
            #     seaborn_config = dict(
            #         rc={"figure.figsize": (11.7, 8.27)},
            #         style="white",
            #     )

            #     log_name = "seaborn_mag_vs_{}_layer-{}".format(coact_type, module.pos)
            #     self.log[log_name] = dict(
            #         data=dataframe,
            #         x=x,
            #         y=y,
            #         # hue=hue,
            #         # style=direction,
            #         # style_order=["Growing", "Shrinking"],
            #         # style=dominant,
            #         # style_order=
            #         # ["11_dominant", "01_dominant", "10_dominant", "00_dominant"],
            #         seaborn_config=seaborn_config,
            #         seaborn_plottype="scatterplot",
            #     )

            # if has_all:

            #     # Convert to factions.
            #     total_coacts = None
            #     for coact_type in coact_types:
            #         print(coact_type)
            #         if total_coacts is None:
            #             total_coacts = coact_d[coact_type].copy()
            #         else:
            #             print('accumulating')
            #             total_coacts[:] += coact_d[coact_type]

            #     for coact_type in coact_types:
            #         coact_d[coact_type] = coact_d[coact_type] / total_coacts

            #     coacts = coact_d.pop("coactivations")
            #     coact_d["coacts_11"] = coacts
            #     dataframe = DataFrame(coact_d)

            #     log_name = "seaborn_coact_pairs_layer-{}".format(module.pos)
            #     self.log[log_name] = dict(
            #         data=dataframe,
            #         seaborn_config=seaborn_config,
            #         seaborn_plottype="pairplot",
            #     )

    def _log_sparse_levels(self):
        with torch.no_grad():
            for idx, module in enumerate(self.model.sparse_modules):
                zero_mask = module.m.weight == 0
                zero_count = torch.sum(zero_mask.int()).item()
                size = np.prod(module.shape)
                log_name = "sparse_level_l" + str(idx)
                self.log[log_name] = 1 - zero_count / size

                # log image as well
                if self.log_sparse_layers_grid:
                    if self.model.has_params(module.m) == "conv":
                        ratio = 255 / np.prod(module.shape[2:])
                        heatmap = (
                            torch.sum(module.m.weight, dim=[2, 3]).float() * ratio
                        ).int()
                        self.log["img_" + log_name] = heatmap.tolist()


class DSNNLogger(SparseLogger):
    def __init__(self, model, config=None):
        super().__init__(model, config)
        defaults = dict(log_surviving_synapses=False, log_masks=False)
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.model = model

    def save_masks(
        self,
        idx,
        new_mask,
        keep_mask,
        add_mask,
        num_add,
        hebbian_mask=None,
        magnitude_mask=None,
    ):
        """Log different masks in DSNN"""

        if self.log_masks:
            num_synapses = np.prod(new_mask.shape)
            self.log["keep_mask_l" + str(idx)] = (
                torch.sum(keep_mask).item() / num_synapses
            )
            self.log["add_mask_l" + str(idx)] = (
                torch.sum(add_mask).item() / num_synapses
            )
            self.log["new_mask_l" + str(idx)] = (
                torch.sum(new_mask).item() / num_synapses
            )
            self.log["missing_weights_l" + str(idx)] = num_add / num_synapses

            # conditional logs
            if hebbian_mask is not None:
                self.log["hebbian_mask_l" + str(idx)] = (
                    torch.sum(hebbian_mask).item() / num_synapses
                )
            if magnitude_mask is not None:
                self.log["magnitude_mask_l" + str(idx)] = (
                    torch.sum(magnitude_mask).item() / num_synapses
                )

    def save_surviving_synapses(self, module, keep_mask, add_mask):
        """Tracks added and surviving synapses"""

        self.survival_ratios = []
        if self.log_surviving_synapses and self.model.pruning_active:
            self.survival_ratios = []
            # count how many synapses from last round have survived
            if module.added_synapses is not None:
                total_added = torch.sum(module.added_synapses).item()
                surviving = torch.sum(module.added_synapses & keep_mask).item()
                if total_added:
                    survival_ratio = surviving / total_added
                    self.survival_ratios.append(survival_ratio)

                # keep track of new synapses to count surviving on next round
                module.added_synapses = add_mask
                self.log["mask_sizes_l" + str(module.pos)] = module.nonzero_params()
                self.log["surviving_synapses_l" + str(module.pos)] = survival_ratio

    def log_post_epoch(self):
        super().log_post_epoch()
        # adds tracking of average surviving synapses
        if self.log_surviving_synapses and self.model.pruning_active:
            self.log["surviving_synapses_avg"] = np.mean(self.survival_ratios)
