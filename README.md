# L1-norm based perturbation CROWN
This project is to integrate the work of [Overcoming the Convex Barrier for Simplex Inputs](https://openreview.net/pdf?id=JXREUkyHi7u) to CROWN algorithm. It cleverly uses the simplex to propose a tighter boundary for the l1 perturbation of the convex activation function network, improving the effect of the CROWN algorithm.

## Quick Start
Use `compute_bounds` interface in `SimplexSolver` class under `src/simplex_solver.py` to compute bounds. We also provided an example in `src/simplex_solver.py`

## References
This work is based on following papers:
```
@article{behl2021overcoming,
  title={Overcoming the convex barrier for simplex inputs},
  author={Behl, Harkirat Singh and Kumar, M Pawan and Torr, Philip and Dvijotham, Krishnamurthy},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={4871--4882},
  year={2021}
}

@article{zhang2018efficient,
  title={Efficient Neural Network Robustness Certification with General Activation Functions},
  author={Zhang, Huan and Weng, Tsui-Wei and Chen, Pin-Yu and Hsieh, Cho-Jui and Daniel, Luca},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  pages={4939--4948},
  year={2018},
  url={https://arxiv.org/pdf/1811.00866.pdf}
}

@article{xu2020automatic,
  title={Automatic perturbation analysis for scalable certified robustness and beyond},
  author={Xu, Kaidi and Shi, Zhouxing and Zhang, Huan and Wang, Yihan and Chang, Kai-Wei and Huang, Minlie and Kailkhura, Bhavya and Lin, Xue and Hsieh, Cho-Jui},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

**Please notice that `src/lirpa` directory is the code for original method provided in [Overcoming the Convex Barrier for Simplex Inputs](https://openreview.net/pdf?id=JXREUkyHi7u). Also `src/crown.py`, `src/linear.py`, `src/relu.py` are based on the code `git@github.com:huanzhang12/ECE584-SP24-assignment2.git` from @huanzhang12 and @schawla7.**