# Neural Temporal Point Process with Transformer
## Introduction
Our goal is to develop neural temporal point proces to capture long-range dependencies among sequentially observed events, using transformer

## Requirements
We offer a toy train data and a toy eval data in the `data` folder

You can generate more data with `tick` lib, we also offer the data-generating code in `HawkesDataset.ipynb`

It is important to note that `tick` require `3.6<=python<=3.8`

Other requirements:
* pytorch >= 1.8
* diffusers >= 2.14.4

## Detail
Hawkes process is a kind of temporal point process, the intensity will decay exponentially after event happen.

Briefly, the intensity (at time $t$) of Hawkes process can be described as such a simplified equation: $\lambda(t) = \mu + \alpha\beta(t)$

$\mu$ is the base intensity, $\alpha$ is the influence of history (maybe from self or other event type), and $\beta(t)$ is the decay function

You can get more detail from the document of [`tick`](https://x-datainitiative.github.io/tick/modules/generated/tick.hawkes.HawkesExpKern.html#tick.hawkes.HawkesExpKern)

In order to model Hawkes process, we build an `encoder-decoder` model: encoder encodes events history to representations, and decoder decodes representations to intensity or cumulative intensity.

In `SA-MLP_Hawkes_1D.ipynb`, we use a self-attention block as encoder, and a MLP block as decoder

This model can be trained via maximize the likelihood function:
$$\ell = \sum_{i:t_i \leqslant T} log \lambda(t_i) - \underbrace{\int_{t=0}^{T} \lambda(t)dt}_{\Lambda}$$
$t_i$ represents the timestamp of $i$-th event

$\Lambda$ can be estimated by Monte Carlo algorithm