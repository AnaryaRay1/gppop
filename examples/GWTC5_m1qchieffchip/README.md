# Example run for $m_1$,$q$,$\chi_{eff}$,$\chi_p$ population inference from GWTC5

For reproducing the results of [Ray and Kalogera, 2025](https://arxiv.org/abs/2607.28622):

Download GWTC5 public data (see ref. above for links).

Update data paths in ```gwtc-5.yaml```

Then run:

```
python prepare_input.py config.yaml

run_gppop --metadata __run__/gppop_metadata.h5 --ntune=3000 --nsteps=3000 --target_accept=0.99 --sigma_sd 1.0 --mc_convergence_check=True --exponent=-75 --maximum_uncertainty=1.0 --njobs=4 --devices cuda:0,cuda:1,cuda:2,cuda:3 --output=__run__/posterior.pkl
```

Plotting code will be uploaded soon.