from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import jax.numpy as jnp
import haiku as hk

from typing import Optional
from score_estimation.matrix_kernels.curlfree_kernel import CurlFreeIMQKernel, CurlFreeSEKernel
from score_estimation.estimators.abstract import AbstractScoreEstimator
from score_estimation.matrix_kernels.utils import random_choice, conjugate_gradient


class Tikhonov(AbstractScoreEstimator):
    def __init__(self,
                 lam: Optional[float] = 1e-1,
                 kernel_type: str = 'curl_free_imq',
                 bandwidth: float = 1.0,
                 truncated_tikhonov=False,
                 subsample_rate=None,
                 use_cg=True,
                 tol_cg=1.0e-4,
                 maxiter_cg=40,
                 key=hk.PRNGSequence(1234)):
        super().__init__()

        if kernel_type == 'curl_free_imq':
            self._kernel = CurlFreeIMQKernel(kernel_hyperparams=bandwidth)
        elif kernel_type == 'curl_free_se':
            self._kernel = CurlFreeSEKernel(kernel_hyperparams=bandwidth)
        else:
            raise NotImplementedError(f'Kernel type {kernel_type} is not implemented.')

        self._lam = lam
        self._use_cg = use_cg
        self._tol_cg = tol_cg
        self._subsample_rate = subsample_rate
        self._maxiter_cg = maxiter_cg
        self._truncated_tikhonov = truncated_tikhonov
        self.bandwidth = bandwidth
        self.name = "Tikhonov" if self._subsample_rate is None else "NKEF"
        self.hk_key = key

    def estimate_gradients_s_x(self, queries: jnp.ndarray, samples: jnp.ndarray) -> jnp.array:
        self.fit(samples)
        score_estimate = self._compute_gradients(queries=queries)
        assert queries.shape == score_estimate.shape
        return score_estimate

    def fit(self, samples: jnp.array):
        if self._subsample_rate is None:
            return self._fit_exact(samples)
        else:
            return self._fit_subsample(samples)

    def _compute_energy(self, samples: jnp.ndarray) -> jnp.ndarray:
        if self._subsample_rate is None and not self._truncated_tikhonov:
            Kxq, div_xq = self._kernel.kernel_energy(samples, self._samples,
                                                     kernel_hyperparams=self._kernel_hyperparams)
            div_xq = jnp.mean(div_xq, axis=-1) / self._lam
            Kxq = jnp.reshape(Kxq, (jnp.shape(samples)[-2], -1))
            energy = jnp.matmul(Kxq, self._coeff)
            energy = jnp.reshape(energy, [-1]) - div_xq
        else:
            Kxq = self._kernel.kernel_energy(samples, self._samples,
                                             kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
            Kxq = jnp.reshape(Kxq, (jnp.shape(samples)[-2], -1))
            energy = -jnp.matmul(Kxq, self._coeff)
            energy = jnp.reshape(energy, [-1])
        return energy

    def _compute_gradients(self, queries: jnp.ndarray) -> jnp.ndarray:
        d = jnp.shape(queries)[-1]
        if self._subsample_rate is None and not self._truncated_tikhonov:
            Kxq_op, div_xq = self._kernel.kernel_operator(queries, self._samples,
                                                          kernel_hyperparams=self._kernel_hyperparams)
            div_xq = jnp.mean(div_xq, axis=-2) / self._lam
            grads = Kxq_op.apply(self._coeff)
            grads = jnp.reshape(grads, (-1, d)) - div_xq
        else:
            Kxq_op = self._kernel.kernel_operator(queries, self._samples,
                                                  kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
            grads = -Kxq_op.apply(self._coeff)
            grads = jnp.reshape(grads, (-1, d))
        return grads

    def _fit_subsample(self, samples: jnp.ndarray):
        if self.bandwidth is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        else:
            kernel_hyperparams = self.bandwidth
        self._kernel_hyperparams = kernel_hyperparams

        M, d = np.shape(samples)[-2], np.shape(samples)[-1]
        N = np.array(np.array(M) * self._subsample_rate).astype(np.int32)

        subsamples = random_choice(samples, N, next(self.hk_key))
        Knn_op = self._kernel.kernel_operator(subsamples, subsamples,
                                              kernel_hyperparams=kernel_hyperparams, compute_divergence=False)
        Knm_op, K_div = self._kernel.kernel_operator(subsamples, samples,
                                                     kernel_hyperparams=kernel_hyperparams)
        self._samples = subsamples

        if self._use_cg:
            def apply_kernel(v):
                return Knm_op.apply(Knm_op.apply_adjoint(v)) / jnp.array(M) \
                       + self._lam * Knn_op.apply(v)

            linear_operator = collections.namedtuple(
                "LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])
            Kcg_op = linear_operator(
                shape=Knn_op.shape,
                dtype=Knn_op.dtype,
                apply=apply_kernel,
                apply_adjoint=apply_kernel,
            )
            H_dh = jnp.mean(K_div, axis=-2)
            H_dh = jnp.reshape(H_dh, (N * d))
            conj_ret = conjugate_gradient(
                Kcg_op, H_dh, max_iter=self._maxiter_cg, tol=self._tol_cg)
            self._coeff = jnp.reshape(conj_ret.x, (N * d, 1))
        else:
            Knn = Knn_op.kernel_matrix(flatten=True)
            Knm = Knm_op.kernel_matrix(flatten=True)
            K_inner = jnp.matmul(Knm, jnp.transpose(Knm)) / jnp.array(M) + self._lam * Knn  # matmul said transpose b
            H_dh = jnp.mean(K_div, axis=-2)

            if self._kernel.kernel_type() == 'diagonal':
                K_inner += 1.0e-7 * jnp.eye(N)
                H_dh = jnp.reshape(H_dh, (N, d))
            else:
                # The original Nystrom KEF estimator (Sutherland et al., 2018).
                # Adding the small identity matrix is necessary for numerical stability.
                K_inner += 1.0e-7 * jnp.eye(N * d)
                H_dh = jnp.reshape(H_dh, (N * d, 1))
            self._coeff = jnp.reshape(jnp.linalg.solve(K_inner, H_dh), (N * d, 1))

    def _fit_exact(self, samples: jnp.ndarray):
        # samples: [M, d]
        if self.bandwidth is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
            # print("Median value is: ", kernel_hyperparams)
        else:
            kernel_hyperparams = self.bandwidth
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = jnp.shape(samples)[-2]
        d = jnp.shape(samples)[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                                                   kernel_hyperparams=kernel_hyperparams)

        if self._use_cg:
            if self._truncated_tikhonov:
                def apply_kernel(v):
                    return K_op.apply(K_op.apply(v) / jnp.array(M) + self._lam * v)
            else:
                def apply_kernel(v):
                    return K_op.apply(v) + jnp.array(M) * self._lam * v

            linear_operator = collections.namedtuple(
                "LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])
            Kcg_op = linear_operator(
                shape=K_op.shape,
                dtype=K_op.dtype,
                apply=apply_kernel,
                apply_adjoint=apply_kernel,
            )
            H_dh = jnp.mean(K_div, axis=-2)
            H_dh = jnp.reshape(H_dh, (M * d)) / self._lam
            conj_ret = conjugate_gradient(
                Kcg_op, H_dh, max_iter=self._maxiter_cg, tol=self._tol_cg)
            self._coeff = jnp.reshape(conj_ret.x, (M * d, 1))
        else:
            K = K_op.kernel_matrix(flatten=True)
            H_dh = jnp.mean(K_div, axis=-2)
            if self._kernel.kernel_type() == 'diagonal':
                identity = jnp.eye(M)
                H_shape = [M, d]
            else:
                identity = jnp.eye(M * d)
                H_shape = [M * d, 1]

            if self._truncated_tikhonov:
                # The Nystrom version of KEF with full samples.
                # See Example 3.6 for more details.
                K = jnp.matmul(K, K) / jnp.array(M) \
                    + self._lam * K + 1.0e-7 * identity
            else:
                # The original KEF estimator (Sriperumbudur et al., 2017).
                K += jnp.array(M) * self._lam * identity
            H_dh = jnp.reshape(H_dh, H_shape) / self._lam
            self._coeff = jnp.reshape(jnp.linalg.solve(K, H_dh), (M * d, 1))
