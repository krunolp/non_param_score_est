import jax
import jax.numpy as jnp
import collections


def median_heuristic(x, y):
    # x: [..., n, d]
    # y: [..., m, d]
    # return: []
    n = jnp.shape(x)[-2]
    m = jnp.shape(y)[-2]
    x_expand = jnp.expand_dims(x, -2)
    y_expand = jnp.expand_dims(y, -3)
    pairwise_dist = jnp.sqrt(jnp.sum(jnp.square(x_expand - y_expand), axis=-1))
    k = n * m // 2
    top_k_values = jax.lax.top_k(
        jnp.reshape(pairwise_dist, [-1, n * m]),
        k=k)[0]
    kernel_width = jnp.reshape(top_k_values[:, -1], jnp.shape(x)[:-2])
    return jax.lax.stop_gradient(kernel_width)


def random_choice(inputs, n_samples, rng_key):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = jnp.expand_dims(jnp.zeros(jnp.shape(inputs)[0]), 0)

    ind = jax.random.categorical(rng_key, uniform_log_prob, shape=(n_samples, 1))
    ind = jnp.squeeze(ind)  # (n_samples,)

    return inputs[ind, :]


def conjugate_gradient(operator,
                       rhs,
                       x=None,
                       tol=1e-4,
                       max_iter=40):
    '''From tensorflow/contrib/solvers/linear_equations.py'''

    cg_state = collections.namedtuple("CGState", ["i", "x", "r", "p", "gamma"])

    def stopping_criterion(param):
        i, state = param
        return jnp.logical_and(i < max_iter, jnp.linalg.norm(state.r) > tol)

    def cg_step(param):
        i, state = param
        z = operator.apply(state.p)
        alpha = state.gamma / jnp.sum(state.p * z)
        x = state.x + alpha * state.p
        r = state.r - alpha * z
        gamma = jnp.sum(r * r)
        beta = gamma / state.gamma
        p = r + beta * state.p
        param_ret = i + 1, cg_state(i + 1, x, r, p, gamma)
        return param_ret

    n = operator.shape[1:]
    rhs = jnp.expand_dims(rhs, -1)
    if x is None:
        x = jnp.expand_dims(jnp.zeros(n), -1)
        r0 = rhs
    else:
        x = jnp.expand_dims(x, -1)
        r0 = rhs - operator.apply(x)

    p0 = r0
    gamma0 = jnp.sum(r0 * p0)
    tol *= jnp.linalg.norm(r0)
    i = jnp.array(0).astype(jnp.int32)
    state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0)
    _, state = jax.lax.while_loop(stopping_criterion, cg_step, (i, state))
    return cg_state(
        state.i,
        x=jnp.squeeze(state.x),
        r=jnp.squeeze(state.r),
        p=jnp.squeeze(state.p),
        gamma=state.gamma)
