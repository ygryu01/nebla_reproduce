
# Stage1: MLP predictor

1) TEST
**--already_normalized: CBCT DATASET이 정규화된 경우에만 사용한다. 그렇지 않은 경우 train.py의 --clip_min/max를 조절하여 정규화를 해야 한다.**

```bash
CUDA_VISIBLE_DEVICES=7 python train_stage1.py \
  --out_dir ./logs/nebla_stage1_mlp \
  --iters 30000 \
  --batch_size 16 \
  --n_points 32768 \
  --n_samples 400 \
  --already_normalized \
  --amp
```

2) INFER
**--overwrite는 이미 존재하는 inference 파일을 덮어씌운다.**
```bash
CUDA_VISIBLE_DEVICES=7 python infer_stage1.py \
  --stage1_ckpt ./logs/nebla_stage1_mlp/ckpt_final.pt \
  --rho_out_root ./logs/nebla_stage1_rho \
  --split all \
  --n_samples 400 \
  --dense_k_stride 1 \
  --dense_chunk_points 131072 \
  --save_dtype float16 \
  --amp
```

저장 형식은 아래와 같다.
./logs/nebla_stage1_rho/
  train/{sid}.pt
  val/{sid}.pt
  test/{sid}.pt

각 파일에는 아래의 내용이 담겨 있다.
```python
{
    "sid": sid,
    "rho":    [1, D, H, W],
    "mask":   [1, D, H, W],
    "target": [D, H, W],
    "meta":   {...}
}
```

# Stage2: 3D U-Net Refiner

1) TEST
```bash
CUDA_VISIBLE_DEVICES=7 python train_stage2.py \
  --rho_root ./logs/nebla_stage1_rho \
  --out_dir ./logs/nebla_stage2_refiner \
  --batch_size 1 \
  --refiner_f_maps 32,64,128,256 \
  --debug_image_every 1000 \
  --debug_image_max_items 2 \
  --debug_image_split val \
  --amp
```

2) INFER
## 주의: INFER 코드에는 metric을 계산하는 부분이 없음.
**--save_input: 3D U-NET에 들어가는 input volume(coarse volume)을 저장한다.**
**--save_target: GT volume을 저장한다.**
```bash
CUDA_VISIBLE_DEVICES=7 python infer_stage2.py \
  --stage2_ckpt ./logs/nebla_stage2_refiner/ckpt_final.pt \
  --rho_root ./logs/nebla_stage1_rho \
  --out_root ./logs/nebla_stage2_infer \
  --split test \
  --batch_size 1 \
  --save_mip_images \
  --save_input \
  --amp
```