# APT-MMF

This repository provides a reference implementation of *APT-MMF* as described in the paper:<br>
> APT-MMF: An advanced persistent threat actor attribution method based on multimodal and multilevel feature fusion.<br>
> *Under Review*, 2024.<br>

## Installation and Execution

### From Source

Start by grabbing this source code:
```
git clone https://github.com/NanArtist/APT-MMF.git
```

### Environment

It is recommended to run this code inside a `conda` environment with `python3.10`.
- Create environment:
   ```sh
   conda create -n APT-MMF python=3.10
   ```
- Activate environment:
   ```sh
   conda activate APT-MMF
   ```

### Requirements

Latest tested combination of the following packages for Python 3 are required:

- PyTorch
- DGL
- NetworkX
- scikit-learn
- NumPy
- SciPy


To install all the requirements, run the following command:

```
python -m pip install -r requirements.txt
```

### Execution
 
Once the environment is configured, the programs can be run by the following command:
   ```sh
    python Main.py
   ```

## Citing

If you find APT-MMF useful in your research, please cite the following paper:

    @article{xiao_apt-mmf_2024,
        title={APT-MMF: An advanced persistent threat actor attribution method based on multimodal and multilevel feature fusion},
        author={Xiao, Nan and Lang, Bo and Wang, Ting and Chen, Yikai},
        journal={arXiv preprint arXiv:2402.12743},
        year={2024}
    }

The preprint version of our paper is available at [arXiv:2402.12743](https://arxiv.org/abs/2402.12743).

*Note: Additional resources will be released publicly after the publication of our paper.*
