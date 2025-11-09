#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn.masking import apply_mask
from nilearn.signal import clean

# === Paths ===
func_path = Path('/BICNAS2/group-northoff/rsfMRI-FEOBV/sub-002/ses-002/func/sub-002_ses-002_task-rest_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
mask_dir = Path('/home/stubanadean/ROIs_NII')
output_dir = Path('/home/stubanadean/voxelwise_timeseries_nilearn')
output_dir.mkdir(parents=True, exist_ok=True)

# === Build confounds filepath ===
base = func_path.name.split('_space-')[0]
confound_file = func_path.parent / f"{base}_desc-confounds_timeseries.tsv"
print(f"Confound file: {confound_file}")

# === Load confounds ===
confounds_df = pd.read_csv(confound_file, sep='\t')

# === Select relevant confound columns ===
# Motion parameters, their derivatives, and quadratic terms (as per fMRIPrep output)
# === Define confound regressors ===
motion_cols = [
    # Motion parameters + derivatives + quadratic terms
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
    'trans_x_power2', 'trans_y_power2', 'trans_z_power2',
    'rot_x_power2', 'rot_y_power2', 'rot_z_power2',
    'trans_x_derivative1_power2', 'trans_y_derivative1_power2', 'trans_z_derivative1_power2',
    'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2'
]

# aCompCor components (first 6, per fMRIPrep recommendations)
acompcor_cols = [f'a_comp_cor_{i:02d}' for i in range(6)]

# Combine all potential regressors
candidate_columns = motion_cols + acompcor_cols

# Keep only those that exist in the confounds file
selected_columns = [col for col in candidate_columns if col in confounds_df.columns]
confounds = confounds_df[selected_columns].values
print("Using confounds columns:", selected_columns)

# === Motion summary for QC ===
if 'framewise_displacement' in confounds_df.columns:
    mean_fd = confounds_df['framewise_displacement'].mean(skipna=True)
    max_fd = confounds_df['framewise_displacement'].max(skipna=True)
    print(f"Mean FD: {mean_fd:.4f}, Max FD: {max_fd:.4f}")

# === Check confound integrity ===
print("Confounds shape:", confounds.shape)
print("NaNs in confounds:", np.isnan(confounds).sum())
print("Infs in confounds:", np.isinf(confounds).sum())

if np.isnan(confounds).any() or np.isinf(confounds).any():
    print("Confounds contain NaNs or infs, cleaning now...")
    confounds = np.nan_to_num(confounds, nan=0.0, posinf=0.0, neginf=0.0)
# === Check confound integrity ===
print("Confounds shape:", confounds.shape)
print("NaNs in confounds:", np.isnan(confounds).sum())
print("Infs in confounds:", np.isinf(confounds).sum())

# === Clean old output directory ===
import shutil
if output_dir.exists():
    print(f"Removing old results in {output_dir}...")
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# === Load masks ===
layers = {
    'Exteroception': [Path("/home/stubanadean/ROIs_NII/Exteroception.nii.gz")],
    'Interoception': [Path("/home/stubanadean/ROIs_NII/Interoception.nii.gz")],
    'Cognition': [Path("/home/stubanadean/ROIs_NII/Cognition.nii.gz")]
}

# === Process masks ===
for layer, mask_paths in layers.items():
    print(f"\nProcessing {layer}")
    layer_output = output_dir / layer
    layer_output.mkdir(exist_ok=True)

    if not mask_paths:
        print(f"  No masks found for {layer}")
        continue

    for mask_path in mask_paths:
        mask_name = mask_path.stem
        print(f"  Mask: {mask_name}")

        try:
            # Extract voxelwise time series (shape: timepoints x voxels)
            masked_ts = apply_mask(str(func_path), str(mask_path))
            print(f"    Extracted time series shape: {masked_ts.shape}")

            # Remove voxels with any NaN or inf values
            valid_voxels = np.all(np.isfinite(masked_ts), axis=0)
            print(f"    Valid voxels count: {np.sum(valid_voxels)} / {masked_ts.shape[1]}")

            if not np.any(valid_voxels):
                print(f"    Warning: All voxels contain NaNs or infs for {mask_name}. Skipping.")
                continue

            cleaned_masked_ts = masked_ts[:, valid_voxels]

            # Denoise time series using confounds
            cleaned_ts = clean(cleaned_masked_ts, confounds=confounds, standardize="zscore_sample")

            # Check for NaNs or infs after cleaning
            print(f"    NaNs in cleaned_ts: {np.isnan(cleaned_ts).sum()}")
            print(f"    Infs in cleaned_ts: {np.isinf(cleaned_ts).sum()}")

            valid_after_clean = np.all(np.isfinite(cleaned_ts), axis=0)

            if not np.any(valid_after_clean):
                print(f"    Warning: All voxels contain NaNs or infs after cleaning for {mask_name}. Skipping.")
                continue

            cleaned_ts = cleaned_ts[:, valid_after_clean]

            # Save cleaned voxelwise time series
            save_path = layer_output / f"{func_path.stem}_{mask_name}_voxelwise_timeseries.npy"
            np.save(save_path, cleaned_ts)
            print(f"    Saved: {save_path.name}")

        except Exception as e:
            print(f"    Error processing {mask_name}: {e}")

print("\n[Done] All voxelwise time series extracted and saved.")
