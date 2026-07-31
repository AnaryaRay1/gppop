# GPpop
Multi-dimensional non-parametric population inference for compact binary mergers using GPU-accelerated binned Gaussian processes.

### Installation

```
conda create -n gppop-env python=3.11
git clone https://github.com/AnaryaRay1/gppop.git
cd gppop
pip install .
```
Example for running analysis for four-dimensional inference in GWTC5: ```examples/GWTC5_m1qchieffchip/```

### Citations:

If you use this package for your publications, please cite the following works:



```
@article{ray2026fourdimensionalmodelagnosticprobeastrophysical,
      title={A Four-dimensional Model-agnostic Probe into the Astrophysical Origins of Binary Black Hole Subpopulations}, 
      author={Anarya Ray and Vicky Kalogera},
      year={2026},
      eprint={2607.28622},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE},
      url={https://arxiv.org/abs/2607.28622}, 
}
@article{Sridhar:2025kvi,
    author = "Sridhar, Omkar and Ray, Anarya and Kalogera, Vicky",
    title = "{Characterizing Binary Black Hole Subpopulations in GWTC-4 with Binned Gaussian Processes: On the Origins of the 35 M$_{⊙}$ Peak}",
    eprint = "2511.22093",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    reportNumber = "LIGO-P2500712",
    doi = "10.3847/2041-8213/ae8011",
    journal = "Astrophys. J. Lett.",
    volume = "1005",
    number = "2",
    pages = "L54",
    year = "2026"
}

@article{Ray:2024hos,
    author = "Ray, Anarya and Maga{\~n}a Hernandez, Ignacio and Breivik, Katelyn and Creighton, Jolien",
    title = "{Searching for Binary Black Hole Subpopulations in Gravitational-wave Data Using Binned Gaussian Processes}",
    eprint = "2404.03166",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    reportNumber = "LIGO-P2400115",
    doi = "10.3847/1538-4357/adf22a",
    journal = "Astrophys. J.",
    volume = "991",
    number = "1",
    pages = "17",
    year = "2025"
}

@article{Ray:2023upk,
    author = "Ray, Anarya and Maga{\~n}a Hernandez, Ignacio and Mohite, Siddharth and Creighton, Jolien and Kapadia, Shasvath",
    title = "{Nonparametric Inference of the Population of Compact Binaries from Gravitational-wave Observations Using Binned Gaussian Processes}",
    eprint = "2304.08046",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P2300098",
    doi = "10.3847/1538-4357/acf452",
    journal = "Astrophys. J.",
    volume = "957",
    number = "1",
    pages = "37",
    year = "2023"
}
```

