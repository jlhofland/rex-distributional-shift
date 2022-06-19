# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adopted from Invariant Risk Minimization
# https://github.com/facebookresearch/InvariantRiskMinimization
# 

import torch


class DataModel(object):
    def __init__(self, dim, shift, ones=True, scramble=False, hetero=True, confounder_on_x=False):
        self.hetero = hetero
        self.dim = dim // 2

        if ones:
            # Invariant relationship from X to Y and Y to Z set to Identity
            self.wxy = torch.eye(self.dim)
            self.wyz = torch.eye(self.dim)
        else:
            # Invariant relationship from X to Y  and Y to Z set to Gaussian
            self.wxy = torch.randn(self.dim, self.dim) / dim + 1
            self.wyz = torch.randn(self.dim, self.dim) / dim + 1

        if scramble:
            self.scramble, _ = torch.linalg.qr(torch.randn(dim, dim))
        else:
            self.scramble = torch.eye(dim)

        if confounder_on_x:
            # X related to confounder
            self.whx = torch.eye(self.dim)
        else:
            # Confounder never relates to X
            self.whx = torch.zeros(self.dim, self.dim)

        if shift == "CS":
            # No relations
            self.why = torch.zeros(self.dim, self.dim)
            self.whz = torch.zeros(self.dim, self.dim)
            self.wyz = torch.zeros(self.dim, self.dim)
        elif shift == "CF":
            # Confounder relates to Y and Z
            self.why = torch.randn(self.dim, self.dim) / dim
            self.whz = torch.randn(self.dim, self.dim) / dim
            self.wyz = torch.zeros(self.dim, self.dim)

            if confounder_on_x:
                self.whx = torch.randn(self.dim, self.dim) / dim

        elif shift == "AC":
            # Y relates to Z
            self.why = torch.zeros(self.dim, self.dim)
            self.whz = torch.zeros(self.dim, self.dim)
        elif shift == "HB":
            # Confounder relates to Y and Z
            # Y relates to Z
            self.why = torch.randn(self.dim, self.dim) / dim
            self.whz = torch.randn(self.dim, self.dim) / dim

            if confounder_on_x:
                self.whx = torch.randn(self.dim, self.dim) / dim

        else:
          raise ValueError("Shift should be CS, CF, AC or HB!")

    def solution(self):
        w = torch.cat((self.wxy.sum(1), torch.zeros(self.dim))).view(-1, 1)
        return w, self.scramble.t()

    def __call__(self, n, env):
        h = torch.randn(n, self.dim) * env
        x = h @ self.whx + torch.randn(n, self.dim) * env

        if self.hetero:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim) * env
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim)
        else:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim) * env

        return torch.cat((x, z), 1) @ self.scramble, y.sum(1, keepdim=True)
