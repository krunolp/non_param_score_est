from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as jnp

from typing import Optional
from non_param_score_est.matrix_kernels.curlfree_kernel import CurlFreeIMQKernel, CurlFreeSEKernel
from non_param_score_est.estimators.abstract import AbstractScoreEstimator


class Landweber(AbstractScoreEstimator):
    def __init__(self,
                 lam: Optional[float] = None,
                 num_iter: Optional[int] = None,
                 kernel_type: str = 'curl_free_imq',
                 bandwidth=None,
                 stepsize=1.0e-2):
        super().__init__()

        self._kernel_hyperparams = None
        self._coeff = None
        self._samples = None
        if lam is not None and num_iter is not None:
            raise RuntimeError('Cannot specify `lam` and `iternum` simultaneously.')
        if lam is None and num_iter is None:
            raise RuntimeError('Both `lam` and `iternum` are `None`.')
        if num_iter is not None:
            lam = 1.0 / jnp.array(num_iter)
        else:
            num_iter = jnp.array(1.0 / lam).astype(jnp.int32) + 1

        if kernel_type == 'curl_free_imq':
            self._kernel = CurlFreeIMQKernel(kernel_hyperparams=bandwidth)
        elif kernel_type == 'curl_free_se':
            self._kernel = CurlFreeSEKernel(kernel_hyperparams=bandwidth)
        else:
            raise NotImplementedError(f'Kernel type {kernel_type} is not implemented.')

        self._lam = lam
        self._stepsize = jnp.array(stepsize)
        self.num_iter = num_iter
        self.bandwidth = bandwidth
        self.name = "Landweber"

    def estimate_gradients_s_x(self, queries: jnp.ndarray, samples: jnp.ndarray) -> jnp.array:
        self.fit(samples)
        score_estimate = self._compute_gradients(queries=queries)
        assert queries.shape == score_estimate.shape
        return score_estimate

    def fit(self, samples: jnp.array):
        if self.bandwidth is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        else:
            kernel_hyperparams = self.bandwidth

        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = jnp.shape(samples)[-2]
        d = jnp.shape(samples)[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                                                   kernel_hyperparams=kernel_hyperparams)

        # H_dh: [Md, 1]
        H_dh = jnp.reshape(jnp.mean(K_div, axis=-2), (M * d, 1))

        def get_next(params):
            t_, c = params
            nc = c - self._stepsize * K_op.apply(c) - (jnp.array(t_)) * self._stepsize ** 2 * H_dh
            return t_ + 1, nc

        t, coeff = jax.lax.while_loop(
            lambda params: params[0] < jnp.array(self.num_iter),
            get_next,
            (1, jnp.zeros_like(H_dh))
        )

        self._coeff = (-jnp.array(t) * self._stepsize, coeff)

    def _compute_energy(self, queries: jnp.ndarray) -> jnp.ndarray:
        Kxq, div_xq = self._kernel.kernel_energy(queries, self._samples,
                                                 kernel_hyperparams=self._kernel_hyperparams)
        Kxq = jnp.reshape(Kxq, (jnp.shape(queries)[-2], -1))
        div_xq = jnp.mean(div_xq, axis=-1) * self._coeff[0]
        energy = jnp.reshape(jnp.matmul(Kxq, self._coeff[1]), (-1)) + div_xq
        return energy

    def _compute_gradients(self, queries: jnp.ndarray) -> jnp.ndarray:
        d = jnp.shape(queries)[-1]
        Kxq_op, div_xq = self._kernel.kernel_operator(queries, self._samples,
                                                      kernel_hyperparams=self._kernel_hyperparams)
        div_xq = jnp.mean(div_xq, axis=-2) * self._coeff[0]
        grads = Kxq_op.apply(self._coeff[1])
        grads = jnp.reshape(grads, (-1, d)) + div_xq
        return grads
