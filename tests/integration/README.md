# Integration tests

These run real VASP and are skipped by default.

    export ASSYST_VASP_BINARY=/cmmc/ptmp/hmai/vasp_compiled/intel_suite/\
    vasp.6.4.3_intelsuite_march_znver4/bin/vasp_std
    pytest tests/integration -v

Must run inside a SLURM allocation, since the command uses `srun`. Requires
`~/.pyiron_vasp_config` pointing at a readable POTCAR library.
