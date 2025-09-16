# LieMap: A Fast and Transparent Gradient-Based Method for Cryo-EM Structure Fitting

## Overview

LieMap is a Python script for fitting high-resolution atomic structures (in PDB or mmCIF format) into lower-resolution cryo-electron microscopy (cryo-EM) density maps (in MRC format).

The method utilizes a fully differentiable, gradient-based optimization pipeline built with PyTorch. Key features include:
- **Lie Algebra Parameterization**: Rigid-body transformations (rotation and translation) are represented using the Lie algebra $\mathfrak{se}(3)$, which provides a minimal, 6-parameter representation that avoids the singularity issues found in Euler angles.
- **Direct Density Score**: The optimization maximizes a direct, real-space cross-correlation score, calculated as the mean density value sampled from the map at the atomic coordinates.
- **Multi-Stage Refinement**: A coarse-to-fine strategy is employed for efficiency and accuracy:
    1.  **Coarse Stage**: A rapid search using only C-alpha atoms and a blurred density map to find the global orientation.
    2.  **Fine Stage**: A longer refinement using all atoms and the original map to achieve a high-precision fit.
- **Transparent & Fast**: The entire process is fast and provides real-time feedback on the optimization progress, including the current density score and RMSD to a reference structure.

## Dependencies

The script requires the following Python libraries:
- `torch`: For tensor operations and automatic differentiation.
- `mrcfile`: For reading and writing MRC density map files.
- `numpy`: For numerical operations.

You can install them using pip:
```bash
pip install torch mrcfile numpy
```

## Usage

The script is run from the command line. Below are the available arguments.

```bash
python fitter_final_version_yue.py [ARGUMENTS]
```

### Arguments

- `--mobile_structure` (required): Path to the mobile atomic structure to be fitted (e.g., `1aon.cif`).
- `--target_map` (required): Path to the target cryo-EM density map (e.g., `EMD-1046.map`).
- `--gold_standard_structure` (required): Path to the gold-standard reference structure for calculating the final RMSD (e.g., `1GRU.cif`).
- `--output` (optional): Path to save the final fitted PDB structure. If not provided, a default name will be generated based on the input file and final RMSD (e.g., `1aon_final_version_rmsd_3.23.pdb`).
- `--sigma_level` (optional): The sigma level used to generate the density map mask. Default is `3.0`.

**Note**: The current version of the script (`fitter_final_version_yue.py`) hardcodes the optimization steps and learning rates for its two stages. Therefore, the command-line arguments `--steps` and `--lr` are defined but currently ignored by the main optimization loop.

## Example

This is the command line used to run the test case of fitting the GroEL structure (1aon) into the EMD-1046 map, using 1GRU as the reference.

```bash
python fitter_final_version_yue.py --mobile_structure 1aon.cif --target_map EMD-1046.map --gold_standard_structure 1GRU.cif
```

Please replace the file names with the actual paths to your files. For example, on our test system, the full command was:
```bash
python /Users/huyue/cryoem/fitter_final_version_yue.py --mobile_structure /Users/huyue/cryoem/1aon.cif --target_map /Users/huyue/cryoem/EMD-1046.map --gold_standard_structure /Users/huyue/cryoem/1GRU.cif
```
