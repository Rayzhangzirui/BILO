# BiLO: Bilevel Local Operator Learning for PDE Inverse Problems

This repository contains the code for the paper "[BiLO: Bilevel Local Operator Learning for PDE Inverse Problems](https://arxiv.org/abs/2404.17789)"


![title](./fig_bilo_schematics.png)

<!-- <img src="./fig_bilo_schematics.jpg" width="200"> -->



## Tutorial
See the [tutorial 1](tutorial_scalar.ipynb) (PyTorch) for example of lear
ning unknown scalar PDE paramters.

See the [tutorial 2](tutorial_vector.ipynb) for example of learning unknown function.

See the [tutorial 3](tutorial_scalar_modulus.ipynb) for example of using modulus and modulus-sym.

## Citing
If you use BiLO, please cite the following paper:

```bibtex
@misc{zhang2024bilo,
      title={BiLO: Bilevel Local Operator Learning for PDE inverse problems}, 
      author={Ray Zirui Zhang and Xiaohui Xie and John Lowengrub},
      year={2024},
      eprint={2404.17789},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Experiments
To run the examples in the paper, use the following command:
```python
python ExpScheduler.py examples.yaml
```
And see the examples.yaml for the configuration of the experiments.

## Dataset
The pretrain datasets, which are numerical solutions generated by Matlab, can be found [here](https://drive.google.com/drive/folders/1_PF3SibVj25a_TAqJz7FBh74dW4nQV9w?usp=sharing). 
