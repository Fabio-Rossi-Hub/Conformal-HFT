# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from torchcp.classification.scores.aps import APS


class Margin(APS):
    """
    Bias reduction through conditional conformal prediction (Löfström et al., 2015)

    paper:https://dl.acm.org/doi/abs/10.3233/IDA-150786
    """
    def __init__(self, ) -> None:
        pass

    def _calculate_single_label(self, probs, label):
        row_indices = torch.arange(probs.size(0), device=probs.device)
        target_prob = probs[row_indices, label].clone()
        probs[row_indices, label] = -1
        second_highest_prob = torch.max(probs, dim=-1).values
        return second_highest_prob - target_prob

    def _calculate_all_label(self, probs):
        _, num_labels = probs.shape
        temp_probs = probs.unsqueeze(1).repeat(1, num_labels, 1)
        indices = torch.arange(num_labels).to(probs.device)
        
        temp_probs[:, indices, indices] = -1

        # torch.max(temp_probs, dim=-1) are the second highest probabilities 
        scores = torch.max(temp_probs, dim=-1).values - probs
        return scores
