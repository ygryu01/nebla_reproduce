# NeBLa 3D U-Net refiner integration

This folder contains the refactored training code with optional NeBLa-style 3D U-Net refinement.

## What changed

- `unet3d_refiner.py`: 3D U-Net refinement model.
- `refinement_ops.py`: scatter/gather/loss utilities.
- `train.py`: adds `--use_3d_refiner` and inserts the refiner into the training path.
- `helpers.py`: validation and checkpoint saving now support the optional refiner.
- `visualization.py`: validation MIP rendering can pass the scattered MLP volume through the refiner before saving MIPs.

## Training path when `--use_3d_refiner` is enabled

```text
SimPX
-> 2D UNET feature map
-> point-conditioned MLP prediction
-> scatter_points_to_volume: [B,N,1] -> [B,1,D,H,W]
-> 3D U-Net refiner
-> loss = point_weight * point MSE + volume_weight * volume MSE + proj_weight * MIP projection MSE
```

The implemented projection loss uses axial, coronal, and sagittal MIP MSE. The paper also includes a perceptual loss; this version does not include perceptual loss because the current project code does not yet define a 3D/per-projection perceptual feature extractor.

## Example

```bash
python train.py \
  --use_3d_refiner \
  --batch_size 1 \
  --refiner_f_maps 64,128,256,512 \
  --refiner_point_weight 1.0 \
  --refiner_volume_weight 1.0 \
  --refiner_proj_weight 10.0
```

For memory debugging, start smaller:

```bash
python train.py \
  --use_3d_refiner \
  --batch_size 1 \
  --refiner_f_maps 16,32,64,128 \
  --val_image_every 0
```
