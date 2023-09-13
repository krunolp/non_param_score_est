import matplotlib.pyplot as plt

from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp

from score_estimation.estimators import *


class plotOneDim:
    def __new__(cls, estimator=None):
        instance = super(plotOneDim, cls).__new__(cls)
        return instance

    def __init__(self, estimator=None):
        key = jax.random.PRNGKey(1234)
        key1, key2 = jax.random.split(key)
        self.loc = jnp.array([0.0])
        self.scale = jnp.array([2.0])
        self.n_samples = 1000
        assert self.loc.shape == self.scale.shape == (1,)

        self.est = estimator
        self.dist, self.dist_name = tfp.distributions.Normal(loc=self.loc, scale=self.scale), "Normal"

        self.x_samples = self.dist.sample(self.n_samples, seed=key1)
        self.x_query = jnp.linspace(start=-10, stop=10, num=self.n_samples).reshape(-1, 1)

        self.logprob = lambda x: (self.dist.log_prob(x).sum(), self.dist.log_prob(x))
        self.plot()

    def score_estimation_x_s(self):
        score_estimate = self.est.estimate_gradients_s_x(self.x_query, self.x_samples)
        true_score, logp = jax.grad(self.logprob, has_aux=True)(self.x_query)
        return score_estimate, true_score, logp

    def plot(self):
        score_estimate, true_score, logp = self.score_estimation_x_s()
        plt.plot(self.x_query, true_score, lw=2, label=r"$\nabla_x \log(x)$")
        plt.plot(self.x_query, score_estimate, lw=2, label=r"$\hat{\nabla}_x \log(x)$")
        plt.plot(self.x_query, logp, lw=2, label=r"$\log(x)$")
        plt.title(self.dist_name + f" distr. with {self.n_samples} samples with {self.est.name} estimator",
                  fontsize=13)
        plt.legend(fontsize=15)
        plt.show(block=True)


class plotTwoDim:
    def __new__(cls, estimator=None):
        instance = super(plotTwoDim, cls).__new__(cls)
        return instance

    def __init__(self, estimator=None):
        key = jax.random.PRNGKey(1234)
        key1, key2 = jax.random.split(key)
        self.loc = jnp.array([0.0, 0.0])
        self.scale_diag = jnp.array([2.0, 2.0])
        self.n_samples = 1000
        self.eta = 0.1
        assert self.loc.shape == self.scale_diag.shape == (2,)

        self.est = estimator

        self.dist = tfp.distributions.MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale_diag)
        self.dist_name = "MV Normal"
        self.x_samples = self.dist.sample(1000, seed=key1)
        self.x1, self.x2 = jnp.meshgrid(jnp.linspace(-5, 5, 10), jnp.linspace(-5, 5, 10))
        self.x_query = jnp.stack([self.x1.flatten(), self.x2.flatten()], axis=-1)
        self.logprob = lambda x: (self.dist.log_prob(x).sum(), self.dist.log_prob(x))
        self.plot()

    def test_score_estimation_x_s(self):
        score_estimate = self.est.estimate_gradients_s_x(self.x_query, self.x_samples)
        true_score, logp = jax.grad(self.logprob, has_aux=True)(self.x_query)
        return score_estimate, true_score, logp

    def plot(self):
        score_estimate, true_score, logp = self.test_score_estimation_x_s()
        plt.quiver(self.x1, self.x2, jnp.reshape(true_score[:, 0], self.x1.shape),
                   jnp.reshape(true_score[:, 1], self.x1.shape), label='True score', color='blue')
        plt.quiver(self.x1, self.x2, jnp.reshape(score_estimate[:, 0], self.x1.shape),
                   jnp.reshape(score_estimate[:, 1], self.x1.shape), label='Estimated score', color='red')
        plt.title(self.dist_name + f" distr. with {self.n_samples} samples with {self.est.name} estimator",
                  fontsize=13)
        plt.legend(fontsize=10)
        plt.show()


if __name__ == '__main__':
    """    To plot the results, choose an estimator from the following list:    """

    # Tikhonov regularization
    est = Tikhonov(bandwidth=10., lam=1e-5)
    #
    # # NKEF
    # est = Tikhonov(bandwidth=10., lam=1e-5, use_cg=False, subsample_rate=0.75)
    #
    # # Landweber iteration
    # est = Landweber(bandwidth=10., lam=1e-5)
    #
    # # nu-method
    # est = NuMethod(bandwidth=10., lam=1e-5)
    #
    # # SSGE
    # est = SSGE(eta=0.1)
    #
    # # Stein estimator
    # est = Stein(lam=1e-5)
    #
    # # KDE estimator
    # est = KDE(bandwidth=1.)

    """    One-dimensional Gaussian distribution experiment   """
    plotOneDim(estimator=est)

    """    Two-dimensional Gaussian distribution experiment   """
    plotTwoDim(estimator=est)
