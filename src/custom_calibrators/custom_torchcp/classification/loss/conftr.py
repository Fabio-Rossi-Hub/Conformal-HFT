# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
__all__ = ["ConfTr"]


import torch

import torch.nn as nn
import torch.nn.functional as F


class ConfTr(nn.Module):
    """
    Conformal Training (Stutz et al., 2021).
    Paper: https://arxiv.org/abs/2110.09192

    :param weight: the weight of each loss function
    :param predictor: the CP predictors
    :param alpha: the significance level for each training batch
    :param fraction: the fraction of the calibration set in each training batch
    :param loss_type: the selected (multi-selected) loss functions, which can be "valid", "classification",  "probs", "coverage".
    :param target_size: Optional: 0 | 1.
    :param loss_transform: a transform for loss
    :param base_loss_fn: a base loss function. For example, cross entropy in classification.
    """
    def __init__(self, weight, predictor, alpha, fraction, loss_type="valid", target_size=1,
                 loss_transform="square", base_loss_fn=None):
        
        super(ConfTr, self).__init__()
        assert weight>0, "weight must be greater than 0."
        assert (0 < fraction < 1), "fraction should be a value in (0,1)."
        assert loss_type in ["valid", "classification",  "probs", "coverage"], ('loss_type should be a value in ["valid", "classification",  "probs", "coverage"].')
        assert target_size==0 or target_size ==1, "target_size should be 0 or 1."
        assert loss_transform in ["square", "abs", "log"], ('loss_transform should be a value in ["square", "abs","log"].')
        self.weight = weight
        self.predictor = predictor
        self.alpha = alpha
        self.fraction = fraction
        self.loss_type =  loss_type
        self.target_size = target_size
        self.base_loss_fn = base_loss_fn
        
        if loss_transform == "square":
            self.transform = torch.square
        elif loss_transform == "abs":
            self.transform = torch.abs
        elif loss_transform == "log":
            self.transform = torch.log
        self.loss_functions_dict = {"valid": self.__compute_hinge_size_loss,
                                    "probs": self.__compute_probabilistic_size_loss,
                                    "coverage": self.__compute_coverage_loss,
                                    "classification": self.__compute_classification_loss
                                    }

    def forward(self, logits, labels):
        # Compute Size Loss
        val_split = int(self.fraction * logits.shape[0])
        cal_logits = logits[:val_split]
        cal_labels = labels[:val_split]
        test_logits = logits[val_split:]
        test_labels = labels[val_split:]

        self.predictor.calculate_threshold(cal_logits.detach(), cal_labels.detach(), self.alpha)
        tau = self.predictor.q_hat
        test_scores = self.predictor.score_function(test_logits)
        # Computing the probability of each label contained in the prediction set.
        pred_sets = torch.sigmoid(tau - test_scores)
        loss = self.weight * self.loss_functions_dict[self.loss_type](pred_sets, test_labels)

        if self.base_loss_fn is not None:
            loss += self.base_loss_fn(logits, labels).float()

        return loss

    def __compute_hinge_size_loss(self, pred_sets, labels):
        return torch.mean(
            self.transform(
                torch.maximum(torch.sum(pred_sets, dim=1) - self.target_size, torch.tensor(0).to(pred_sets.device))))

    def __compute_probabilistic_size_loss(self, pred_sets, labels):
        classes = pred_sets.shape[0]
        one_hot_labels = torch.unsqueeze(torch.eye(classes).to(pred_sets.device), dim=0)
        repeated_confidence_sets = torch.repeat_interleave(
            torch.unsqueeze(pred_sets, 2), classes, dim=2)
        loss = one_hot_labels * repeated_confidence_sets + \
               (1 - one_hot_labels) * (1 - repeated_confidence_sets)
        loss = torch.prod(loss, dim=1)
        return torch.sum(loss, dim=1)

    def __compute_coverage_loss(self, pred_sets, labels):
        one_hot_labels = F.one_hot(labels, num_classes=pred_sets.shape[1])

        # Compute the mean of the sum of confidence_sets multiplied by one_hot_labels
        loss = torch.mean(torch.sum(pred_sets * one_hot_labels, dim=1)) - (1 - self.alpha)

        # Apply the transform function (you need to define this)
        transformed_loss = self.transform(loss)

        return transformed_loss

    def __compute_classification_loss(self, pred_sets, labels):
        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes=pred_sets.shape[1]).float()
        loss_matrix = torch.eye(pred_sets.shape[1], device=pred_sets.device)
        # Calculate l1 and l2 losses
        l1 = (1 - pred_sets) * one_hot_labels * loss_matrix[labels]
        l2 = pred_sets * (1 - one_hot_labels) * loss_matrix[labels]

        # Calculate the total loss
        loss = torch.sum(torch.maximum(l1 + l2, torch.zeros_like(l1, device=pred_sets.device)), dim=1)

        # Return the mean loss
        return torch.mean(loss)
