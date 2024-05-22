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

- PyTorch (2.0.0)
- DGL (1.0.2)
- NetworkX (2.8.4)
- Sklearn (1.2.2)
- NumPy (1.23.5)
- SciPy (1.10.1)


To install all the requirements, run the following command:

```
python -m pip install -r requirements.txt
```

### Execution
 
Once the environment is configured, the programs can be run by the following command:
   ```sh
    python Main.py
   ```

## Introductions
APT-MMF addresses the insufficient feature extraction and fusion problems encountered in the Cyber Threat Intelligence (CTI)-based APT actor attribution research. The main idea is the multimodal and multilevel feature fusion by multimodal node features and multilevel heterogeneous graph attention networks. This repository provides a reference implementation of APT-MMF, including the main programs, various utilities, etc. The execution results of APT-MMF for multiclassification tasks concerning APT actor attribution achieve a Micro-F<sub>1</sub> value of 83.2% and a Macro-F<sub>1</sub> value of 70.5% on a heterogeneous attributed graph dataset contained 1300 APT reports of 21 APT groups.


Please read our paper for more details.
The preprint version of the paper is available at [arXiv:2402.12743](https://arxiv.org/abs/2402.12743).

## Citing

If you find APT-MMF useful in your research, please cite the following paper:

    @article{xiao_apt-mmf_2024,
        title={APT-MMF: An advanced persistent threat actor attribution method based on multimodal and multilevel feature fusion},
        author={Xiao, Nan and Lang, Bo and Wang, Ting and Chen, Yikai},
        journal={arXiv preprint arXiv:2402.12743},
        year={2024}
    }

*Note: Additional resources will be released publicly after the publication of our paper.*
