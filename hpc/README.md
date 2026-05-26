# HPC Execution

Example SLURM scripts are provided in `hpc/slurm/`.

Before submission:

1. Upload this repository to the target cluster.
2. Place raw data in the expected local paths.
3. Edit `PROJECT_ROOT`, partition names, environment activation, and CPU/GPU counts in the SLURM file.
4. Submit with `sbatch`.

The provided templates are examples and may require adjustment for a specific cluster.
