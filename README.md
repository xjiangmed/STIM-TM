# Token Merging via Spatiotemporal Information Mining for Surgical Video Understanding

## STIM-TM applied to Surgformer
In this repository, we provide code for applying Spatiotemporal Information Mining Token Merging (STIM-TM) on the [Surgformer](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_57) baseline.

The provided code extends [the original code for Surgformer](https://github.com/isyangshu/Surgformer).

## Installation
```bash
conda create -n STIM-TM python==3.8.13
conda activate STIM-TM
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Training Surgformer
Please follow the Surgformer code to prepare the dataset and download the pre-trained parameters for TimeSformer.

run the following code for training
```bash
sh scripts/train.sh
```

## Testing Surgformer w/o token merging
1. run the following code for testing, and get **0.txt**, **1.txt**, ... (Testing with N GPUs will result in N files);

```shell
sh scripts/test.sh
```

2. Merge the files and generate separate txt file for each video;
```python
python datasets/convert_results/convert_cholec80.py
python datasets/convert_results/convert_autolaparo.py
```

3. Use [Matlab Evaluation Code](https://github.com/isyangshu/Surgformer/tree/master/evaluation_matlab) to compute metrics;

## Testing Surgformer w/ STIM-TM
run the following code for testing:

```shell
sh scripts/test_STIM_TM.sh
```

## Acknowledgements
Thanks to the authors of following open-source projects:
- [Surgformer](https://github.com/isyangshu/Surgformer)
- [TAPIS](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS)
- [EndoFM-LV](https://github.com/med-air/EndoFM-LV)
- [ToMe](https://github.com/facebookresearch/ToMe)
- [Testa](https://github.com/RenShuhuai-Andy/TESTA)

