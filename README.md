## Mixup for Chaoyang dataset

This repository contains the implementation of mixup on the Chaoyang dataset.

The main idea of *mixup* is to enforce a linear interpolation in both the sample and label spaces:

$$
    x = \lambda x_{1} + (1 - \lambda) x_{2} \quad \text{and} \quad y = \lambda y_{1} + (1 - \lambda) y_{2},
$$

where $\lambda \sim \mathrm{Beta}(\alpha, \beta)$. In the *mixup* paper: $\alpha = \beta$.

A naive implementation did not work because the loss and accuracy prediction seemed to get stuck at certain values and stopped decreasing. According to the official implementation of *mixup*, the loss function can be re-factored as follows:

$$
    \ell_{\mathrm{CE}}(f(x; \theta), y) = -y^{\top} \ln f(x; \theta) = - \left[ \lambda y_{1} + (1 - \lambda) y_{2} \right]^{\top} \ln f(x; \theta) = \lambda \ell_{\mathrm{CE}}(f(x; \theta), y_{1}) + (1 - \lambda) \ell_{\mathrm{CE}}(f(x; \theta), y_{2}).
$$

Thus, the loss is also a convex combination, but evaluated on the same *mixup* samples. The re-factored loss allowed to train *mixup* smoothly. This might need further investigation.

### Results

The evaluation on Chaoyang dataset with ResNet-18 does not show any improvement obtained from *mixup* compared to a conventional classification without *mixup*.