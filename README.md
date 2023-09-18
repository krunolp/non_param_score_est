# Non-parametric score estimation

This is a repository for the [non_param_score_estim](https://pypi.org/project/non-param-score-est/) Python package.

## Usage

### Initial setup

To install the package, you are required to have a Python 3.10 or newer environment. Then, simply run:

 ```
pip install non_param_score_est
 ```

### Choosing the estimator

The following estimators are available (and the corresponding import names):


| Estimator                         | Import Name                   |
|-----------------------------------|:------------------------------|
| Tikhonov regularization           | Tikhonov                      |
| NKEF (with rate 0.75)             | Tikhonov(subsample_rate=0.75) |
| Kernel density estimator          | KDE                           |
| Landweber iteration               | Landweber                     |
| Nu-method                         | NuMethod                      |
| Spectral Stein gradient estimator | SSGE                          |
| Stein estimator                   | Stein                         |

### Utilising the estimators

To use the estimators in your code, simply import the estimator and call the `estimate_gradients_x_s` or `estimate_gradients_s` function. For example, to use the Tikhonov estimator, you would write:
 ```
 import numpy as np
import non_param_score_est
from non_param_score_est.estimators import Tikhonov

samples = np.random.normal(1000)
est = Tikhonov(bandwidth=1., lam=1e-4)

score_estimate = est.estimate_gradients_s(samples)
 ```

## Contributing

We welcome contributions! Please follow these guidelines if you'd like to contribute to the project:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure that tests pass.
4. Submit a pull request with a clear title and description.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The code in [JAX](https://github.com/google/jax) was inspred by the [repository](https://github.com/miskcoo/kscore.git) of the [Nonparametric Score Estimators](https://arxiv.org/abs/2005.10099) paper, by Yuhao Zhou, Jiaxin Shi, Jun Zhu. 

## Contact
Krunoslav Lehman Pavasovic
Email: krunolp@gmail.com
GitHub: krunolp