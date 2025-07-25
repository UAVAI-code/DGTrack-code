# DGTrack
The official implementation for the DGTrack


## Usage
### Installation
Create and activate a conda environment:
```
conda create -n DGTrack python=3.8
conda activate DGTrack
```

Install the required packages:
```
pip install -r requirement.txt
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- UAVDT
        -- VisDrone
   ```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_DGTrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Download pre-trained [DeiT-Tiny-distilled weights](https://github.com/facebookresearch/deit) and put it under `$USER_ROOT$/.cache/torch/hub/checkpoints/. 
```
python tracking/train.py --script dgtrack --config deit_tiny_patch16_224 --save_dir ./output --mode single
```

### Multi-level transformer
```
python  mmdet\datasets\pipelines\Multi_level_run.py
```


### Testing
Download the model weights from [BaiduNetDisk](https://pan.baidu.com/s/1rhHj_ZLqGjT7z27rdozKlQ?pwd=eifm (eifm))

Put the downloaded weights on `<PATH_of_DGTrack>/output/checkpoints/train/DGtrack/deit_tiny_distilled_patch16_224`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

 Testing examples:
- UAVDT
```
python tracking/test.py DGtrack deit_tiny_distilled_patch16_224 --dataset uavdt --threads 4 --num_gpus 1
python tracking/analysis_results.py # need to modify tracker configs and names
```

### Test FLOPs
```
# Profiling DGTrack
python tracking/profile_model.py --script DGtrack --config deit_tiny_distilled_patch16_224
```


