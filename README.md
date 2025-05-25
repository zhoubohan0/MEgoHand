# MEgoHand: Multimodal Egocentric Hand-Object Interaction Motion Generation

<div align="center">

[[Website]](https://zhoubohan0.github.io/MEgoHand/)
[[arXiv]](https://arxiv.org/abs/2505.16602)

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![GitHub license](https://img.shields.io/badge/MIT-blue)]()

![](docs/assets/head.png)

</div>


We propose MEgoHand, a multimodal framework that synthesizes physically
plausible hand-object interactions from egocentric RGB, text, and initial hand pose. 

- High-level "cerebrum": a vision language model (VLM) to infer motion priors from visual-textual context and a monocular depth estimator for object-agnostic spatial reasoning, while
- Low-level "cerebellum": a DiT-based policy generates fine-grained trajectories via flow-matching, along with temporal orthogonal filtering to enhance decoding smoothness.
- Dataset curation: Inverse MANO Retargeting Network and Virtual RGB-D Renderer are designed to unify diverse HOI datasets.

More Visualization can be found on our [[Website]](https://zhoubohan0.github.io/MEgoHand/).


## Code
We will release our code and part of our dataset soon.

## Citation
If you find our work useful, please consider citing us!
```
@article{zhou2025megohand,
  title={MEgoHand: Multimodal Egocentric Hand-Object Interaction Motion Generation},
  author={Zhou, Bohan and Zhan, Yi and Zhang, Zhongbin and Lu, Zongqing},
  journal={arXiv preprint arXiv:2505.16602},
  year={2025}
}
```
