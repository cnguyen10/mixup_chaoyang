## Mixup for Chaoyang dataset

This repository contains the implementation of mixup on the Chaoyang dataset.

The main idea of *mixup* is to enforce a linear interpolation in both the sample and label spaces:
\[
    \begin{aligned}
    x & \gets \lambda x_{1} + (1 - \lambda) x_{2} \\
    y & \gets \lambda y_{1} + (1 - \lambda) y_{2},
    \end{aligned}
\]
where $\lambda \sim \operatorname{Beta}(\alpha, \beta)$. In the *mixup* paper: $\alpha = \beta$.

A naive implementation would not work because the loss and accuracy prediction seem to get stuck. According to the official implementation of *mixup*, the loss function can be re-factored as follwos:
\[
    \ell_{\mathrm{CE}}(f(x; \theta), y) = -y^{\top} \ln f(x; \theta) = - \left[ \lambda y_{1} + (1 - \lambda) y_{2} \right]^{\top} \ln f(x; \theta) = \lambda \ell_{\mathrm{CE}}(f(x; \theta), y_{1}) + (1 - \lambda) \ell_{\mathrm{CE}}(f(x; \theta), y_{2}).
\]

Thus, the loss is also a convex combination, but evaluated on the same *mixup* samples.

### Results

The evaluation on Chaoyang dataset with ResNet-18 does not show any improvement obtained from *mixup compared to a conventional classification without *mixup*.