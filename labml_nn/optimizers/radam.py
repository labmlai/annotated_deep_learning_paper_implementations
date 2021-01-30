"""
---
title: Rectified Adam (RAdam) optimizer
summary: A simple PyTorch implementation/tutorial of RAdam optimizer.
---

# Rectified Adam (RAdam) optimizer

This implementation is based on
[the official implementation](https://github.com/LiyuanLucasLiu/RAdam)
of the paper
[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265).

We have implemented it in [PyTorch](https://pytorch.org)
as an extension to [our AMSGrad implementation](amsgrad.html)
thus requiring only the modifications to be implemented.

Adam optimizer sometimes converges to a bad local optima during the initial stages of the training;
especially when training transformers.
Researches use warmups to counter this; for the the initial training steps (warm-up stage)
they use a low learning rate.
This paper identifies the problem to be the high variance of adaptive learning rate
during initial stages of training, and counters it using a new rectification term to
reduce variance.

The paper also evaluates two variance reduction mechanisms:
* **Adam-2k**: Only compute the adaptive learning rate ($v_t$ in [Adam](adam.html)) during the first 2k steps,
without changing parameters or calculating momentum ($m_t$).
* **Adam-eps**: Adam with large $\epsilon \approx 10^{-4}$.

## Rectified Adam

Let $\sigma(g_1, ..., g_t)$ and $\psi(g_1, ..., g_t)$ be the functions to calculate
momentum and adaptive learning rate.
For Adam, they are
\begin{align}
\sigma(g_1, ..., g_t) &=  \frac{(1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i}{1 - \beta_1^t} \\
\psi(g_1, ..., g_t) &=  \sqrt \frac{1 - \beta_2^t}{(1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i} g_i^2}
\end{align}

### Exponential moving average as simple moving average

The distribution of exponential moving average can be approximated as a simple moving average.
\begin{align}
p\Bigg(\frac{(1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} g_i^2}{1 - \beta_2^t} \Bigg) \approx
p\Bigg(\frac{\sum_{i=1}^{f(t,\beta_2)} g_{t+1-i}^2}{f(t,\beta_2)} \Bigg)
\end{align}
Here we are taking the simple moving average of the last $f(t,\beta_2)$ gradients.
$f(t,\beta_2)$ satisfies the following,
\begin{align}
\frac{(1-\beta_2) \sum_{i=1}^t \beta_2^{t-i} \cdot i}{1 - \beta_2^t} =
\frac{\sum_{i=1}^{f(t,\beta_2)} (t+1-i)}{f(t,\beta_2)}
\end{align}
which gives,
$$f(t,\beta_2) = \frac{2}{1-\beta_2} - 1 - \frac{2 t \beta_2^t}{1 - \beta_2^t}$$

### Scaled inverse chi-squared

From above we have
$$
p\Big( \psi^2(g_1, ..., g_t) \Big) \approx
p\Bigg(\frac{\sum_{i=1}^{f(t,\beta_2)} g_{t+1-i}^2}{f(t,\beta_2)} \Bigg)
$$
where $g_i \sim \mathcal{N}(0, \sigma^2)$.
Note that $sigma$ here is the standard deviation and different from $\sigma(.)$ for momentum.

[Scaled inverse chi-squared](https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution)
is the distribution of squared inverse of mean of $p$ normal distributions.
$$
p\Bigg(\frac{\sum_{i=1}^{f(t,\beta_2)} g_{t+1-i}^2}{f(t,\beta_2)} \Bigg)
\sim
\text{Scale-inv} \mathcal{X}^2(\rho,\frac{1}{\sigma^2})
$$
where $\rho = f(t,\beta_2)$.

### Rectification

They prove that variance of $\psi(.)$ decreases with $\rho$ when
$\psi^2(.) \sim \text{Scale-inv} \mathcal{X}^2(\rho,\frac{1}{\sigma^2})$.

Therefore the variance is minimized at maximal $\rho$ which is
$\rho_{\infty} = \frac{2}{1-\beta_2} - 1$. Let the minimum variance be $C_{\text{var}}$

In order to ensure that the adaptive learning
rate $\psi(.)$ has consistent variance, we rectify the variance with $r$
\begin{align}
r = \sqrt{\frac{C_{\text{var}}}{Var\big[\psi(.)\big]}}
\end{align}

### Approximating $Var[\psi(.)]$

They estimate $Var[\psi(.)] \approx \frac{Var[\psi^2(.)]}{4 \mathbb{E}[\psi^2(.)}$
based on first order expansion of $\sqrt{\psi^2(.)}$
🤪 I didn't get how it was derived.

From $\text{Scale-inv} \mathcal{X}^2$ distribution we have,
\begin{align}
\mathbb{E}\big[\psi^2(.)\big] &= \frac{\rho / \sigma^2}{\rho-2} \\
Var\big[\psi^2(.)\big] &= \frac{2 \rho / \sigma^4}{(\rho-2)^2 (\rho - 2)}
\end{align}
which gives,
$$
Var[\psi(.)] \approx \frac{\rho}{2(\rho-2)(\rho-4)\sigma^2}
$$

### Rectification term

We have
\begin{align}
r &= \sqrt{\frac{C_{\text{var}}}{Var\big[\psi(.)\big]}} \\
Var[\psi(.)] &\approx \frac{\rho}{2(\rho-2)(\rho-4)\sigma^2}
\end{align}

where $C_{\text{var}}$ is $Var\big[\psi(.)\big]$ for $\rho_\infty$.
Lt $\rho$ and step $t$ be $\rho_t$, and $r_t$ be the rectification term
at step $t$.

\begin{align}
C_{\text{var}} &\approx \frac{\rho_\infty}{2(\rho_\infty-2)(\rho_\infty-4)\sigma^2} \\
Var[\psi(g_1,...,g_t)] &\approx \frac{\rho_t}{2(\rho_t-2)(\rho_t-4)\sigma^2}
\end{align}

This gives,
\begin{align}
r_t &= \sqrt{\frac{(\rho_t-2)(\rho_t-4)\rho_\infty}{(\rho_\infty-2)(\rho_\infty-4)\rho_t}}
\end{align}
"""

import math
from typing import Dict, Optional

import torch

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad


class RAdam(AMSGrad):
    """
    ## Rectified Adam Optimizer

    This class extends from AMSAdam optimizer defined in [`amsadam.py`](amsadam.html).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 amsgrad=False,
                 degenerated_to_sgd=True, defaults=None):
        """
        ### Initialize the optimizer

        * `params` is the list of parameters
        * `lr` is the learning rate $\alpha$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\hat{\epsilon}$ or $\epsilon$ based on `optimized_update`
        * `weight_decay` is an instance of class `WeightDecay` defined in [`__init__.py`](index.html)
        * 'optimized_update' is a flag whether to optimize the bias correction of the second moment
          by doing it after adding $\epsilon$
        * `amsgrad` is a flag indicating whether to use AMSGrad or fallback to plain Adam
        * `degenerate_to_sgd` whether to use sgd when the rectification term $r_t is intractable.
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `RAdam`.
        """
        self.degenerated_to_sgd = degenerated_to_sgd
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, amsgrad, defaults)

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        """
        ### Take an update step for a given parameter tensor

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor  $g_t$ for the parameter $\theta_{t-1}$
        * `param` is the parameter tensor $\theta_{t-1}$
        """

        # Calculate weight decay
        grad = self.weight_decay(param, grad, group)

        # Get $m_t$ and $v_t$; i.e. $\sigma(.)$ and $\psi(.)$ without bias correction
        m, v = self.get_mv(state, group, grad)

        # Calculate $t$ the number of optimizer steps
        state['step'] += 1

        # Perform *RAdam* update
        self.r_adam_update(state, group, param, m, v)

    @staticmethod
    def calc_rectification_term(beta2: float, step: int) -> Optional[float]:
        """
        ### Calculate rectification term $r_t$
        """

        # $\beta_2^t$
        beta2_t = beta2 ** step
        # $$\rho_\infty = \frac{2}{1 - \beta_2} - 1$$
        rho_inf = 2 / (1 - beta2) - 1
        # $$\rho_t = \frac{2}{1-\beta_2} - 1 - \frac{2 t \beta_2^t}{1-\beta_2^t}$$
        rho = rho_inf - 2 * step * beta2_t / (1 - beta2_t)

        # $r_t$ is tractable when $\rho_t >= 4$.
        # We are being a little more conservative since it's an approximated value
        if rho >= 5:
            # $$r_t = \sqrt{\frac{(\rho_t-2)(\rho_t-4)\rho_\infty}{(\rho_\infty-2)(\rho_\infty-4)\rho_t}}$$
            r2 = (rho - 4) / (rho_inf - 4) * (rho - 2) / rho * rho_inf / (rho_inf - 2)
            return math.sqrt(r2)
        else:
            return None

    def r_adam_update(self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter,
                      m: torch.Tensor, v: torch.Tensor):
        """
        ### Do the *RAdam* parameter update

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `param` is the parameter tensor $\theta_{t-1}$
        * `m` and `v` are the uncorrected first and second moments $m_t$ and $v_t$;
          i.e. $\sigma(.)$ and $\psi(.)$ without bias correction
        """

        # Get $\beta_1$ and $\beta_2$
        beta1, beta2 = group['betas']
        # Bias correction term for $\hat{m}_t$, $1 - \beta_1^t$
        bias_correction1 = 1 - beta1 ** state['step']
        # Bias correction term for $\hat{v}_t$, $1 - \beta_2^t$
        bias_correction2 = 1 - beta2 ** state['step']

        r = self.calc_rectification_term(beta2, state['step'])

        # Get learning rate
        lr = self.get_lr(state, group)

        # If $r_t$ is intractable
        if r is not None:
            # Whether to optimize the computation by combining scalar computations
            if self.optimized_update:
                # Denominator $\sqrt{v_t} + \hat{\epsilon}$
                denominator = v.sqrt().add_(group['eps'])
                # Step size $\alpha \sqrt{r_t} * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$
                step_size = lr * math.sqrt(bias_correction2) * r / bias_correction1
                # Update parameters $\theta_t \leftarrow \theta_{t-1} - \alpha \sqrt{r_t} \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \cdot
                #  \frac{m_t}{\sqrt{v_t} + \hat{\epsilon}}$
                param.data.addcdiv_(m, denominator, value=-step_size)
            # Computation without optimization
            else:
                # Denominator  $\frac{\sqrt{v_t}}{\sqrt{1-\beta_2^t}} + \epsilon$
                denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # Step size $\frac{\alpha \sqrt{r_t}}{1-\beta_1^t}$
                step_size = lr * r / bias_correction1
                # Update parameters $\theta_t \leftarrow \theta_{t-1} - \alpha \sqrt{r_t} \cdot
                # \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
                param.data.addcdiv_(m, denominator, value=-step_size)

        # If $r_t$ is intractable do a SGD with momentum
        elif self.degenerated_to_sgd:
            # Step size $\frac{\alpha}{1-\beta_1^t}$
            step_size = lr / bias_correction1
            # Update parameters
            # $\theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat{m}_t$
            param.data.add_(m, alpha=-step_size)


def _test_rectification_term():
    """
    ### Plot $r_t$ against $t$ for various $\beta_2$

    ![Plot of r_t](radam_r_t.png)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    beta2 = [0.9999, 0.999, 0.99, 0.9, 0.8, 0.6, 0.5]
    plt.plot(np.arange(1, 5_000), [[RAdam.calc_rectification_term(b, i) for b in beta2] for i in range(1, 5_000)])
    plt.legend(beta2)
    plt.title("Optimizer")
    plt.show()


if __name__ == '__main__':
    _test_rectification_term()
