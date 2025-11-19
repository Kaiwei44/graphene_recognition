import numpy as np
import torch


class ImprovedMultivariateNormal:
    LOG2PI = float(np.log(2 * np.pi))

    def __init__(
        self,
        loc: torch.Tensor,
        cov: torch.Tensor,
    ):
        self.loc = loc
        batch_shape = torch.broadcast_shapes(cov.shape[:-2], loc.shape[:-1])
        self.loc = self.loc.expand(batch_shape + (-1,))

        self.cov = cov
        self.dim = loc.shape[-1]
        self.norm_LOG2PI = 16 * self.LOG2PI

        self.cholesky: torch.Tensor = torch.linalg.cholesky(cov)
        self.inv_cholesky: torch.Tensor = torch.linalg.inv(self.cholesky)

        self.half_log_det = self.cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def squared_mh_distance(
        self, diff: torch.Tensor, inv_cholesky: torch.Tensor
    ) -> torch.Tensor:
        # Multiply with the inverse Cholesky matrix and sum along the last dimension
        mh_dist_squared = (torch.einsum("kji,nki->nkj", inv_cholesky, diff) ** 2).sum(
            dim=-1
        )

        return mh_dist_squared

    def mh_distance(self, value):
        diff = value - self.loc
        return torch.sqrt(self.squared_mh_distance(diff, self.inv_cholesky))

    def log_prob(self, value):
        diff = value - self.loc
        sq_mh_distance = self.squared_mh_distance(diff, self.inv_cholesky)
        return -0.5 * (self.norm_LOG2PI + sq_mh_distance) - self.half_log_det


class ImprovedMultivariateNormalHalf:
    LOG2PI = float(np.log(2 * np.pi))

    def __init__(
        self,
        loc: torch.Tensor,
        cov: torch.Tensor,
    ):
        self.loc = loc.half()
        batch_shape = torch.broadcast_shapes(cov.shape[:-2], loc.shape[:-1])
        self.loc = self.loc.expand(batch_shape + (-1,))

        self.cov = cov
        self.dim = loc.shape[-1]
        self.norm_LOG2PI = 16 * self.LOG2PI

        self.cholesky: torch.Tensor = torch.linalg.cholesky(cov)
        self.inv_cholesky: torch.Tensor = torch.linalg.inv(self.cholesky).half()

        self.half_log_det = (
            self.cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1).half()
        )

    def squared_mh_distance(
        self, diff: torch.Tensor, inv_cholesky: torch.Tensor
    ) -> torch.Tensor:
        # Multiply with the inverse Cholesky matrix and sum along the last dimension
        mh_dist_squared = (torch.einsum("kji,nki->nkj", inv_cholesky, diff) ** 2).sum(
            dim=-1
        )

        return mh_dist_squared

    def mh_distance(self, value):
        value = value.half()
        diff = value - self.loc
        return torch.sqrt(self.squared_mh_distance(diff, self.inv_cholesky))

    def log_prob(self, value):
        value = value.half()
        diff = value - self.loc
        sq_mh_distance = self.squared_mh_distance(diff, self.inv_cholesky)
        return -0.5 * (self.norm_LOG2PI + sq_mh_distance) - self.half_log_det
