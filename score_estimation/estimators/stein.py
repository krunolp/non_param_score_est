from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as jnp

from score_estimation.matrix_kernels.diagonal_kernel import DiagonalIMQKernel
from score_estimation.estimators.abstract import AbstractScoreEstimator


class Stein(AbstractScoreEstimator):
    def __init__(self,
                 lam: float = 1e-1,
                 kernel_type='diagonal_imq',
                 bandwidth=None):

        self._kernel_hyperparams = None
        self._samples = None
        self._coeff = None

        # TODO: Implement curl-free kernels
        if 'diagonal' not in kernel_type.split('_'):
            raise NotImplementedError('Only support diagonal kernels.')
        else:
            self._kernel = DiagonalIMQKernel(kernel_hyperparams=bandwidth)

        super().__init__()

        self.bandwidth = bandwidth
        self.name = "Stein"
        self._lam = lam

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
        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                                                   kernel_hyperparams=kernel_hyperparams)
        K = K_op.kernel_matrix(flatten=True)
        Mlam = jnp.array(M) ** 2 * self._lam
        Kinv = jnp.linalg.inv(K + Mlam * jnp.eye(M))
        H_dh = jnp.sum(K_div, axis=-2)
        grads = -jnp.matmul(Kinv, H_dh)
        self._coeff = {'Kinv': Kinv, 'grads': grads, 'Mlam': Mlam}

    def _compute_gradients_one(self, queries: jnp.ndarray):
        # Section 3.4 in Li & Turner (2018), the out-of-sample extension.
        Kxx = self._kernel.kernel_matrix(queries, queries,
                                         kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
        Kqx, Kqx_div = self._kernel.kernel_matrix(self._samples, queries,
                                                  kernel_hyperparams=self._kernel_hyperparams)
        KxqKinv = jnp.matmul(jnp.transpose(Kqx), self._coeff['Kinv'])
        term1 = -1. / (Kxx + self._coeff['Mlam'] - jnp.matmul(KxqKinv, Kqx))
        term2 = jnp.matmul(jnp.transpose(Kqx), self._coeff['grads']) \
                - jnp.matmul(KxqKinv + 1., jnp.squeeze(Kqx_div, -2))
        return jnp.matmul(term1, term2)

    def _compute_gradients(self, queries: jnp.ndarray) -> jnp.ndarray:
        if queries is self._samples:
            return self._coeff['grads']
        else:
            def stein_dlog(y):
                stein_dlog_qx = self._compute_gradients_one(jnp.expand_dims(y, 0))
                stein_dlog_qx = jnp.squeeze(stein_dlog_qx, axis=-2)
                return stein_dlog_qx

            return jax.lax.map(stein_dlog, queries)
