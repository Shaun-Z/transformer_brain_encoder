## Transformer brain encoders explain human high-level visual responses

This repository now uses a single config-driven runtime under `src/` for training, evaluation, prediction, and wrapper-based inference. The default paper-faithful preset is:

- `configs/tben/dinov2_q_rois_all.yaml`

## Model architecture

<img src="https://raw.githubusercontent.com/Hosseinadeli/transformer_brain_encoder/main/figures/arch.png" width = 1000>

The paper path is:

1. load NSD images from `/data/algonauts_2023_challenge_data`
2. encode them with a frozen backbone
3. route retinotopic token features through ROI queries in a transformer decoder
4. predict left and right hemisphere responses

## Inspect the dataset

```bash
uv run python -m src.cli.inspect_data --dataset-root /data/algonauts_2023_challenge_data --subject 1
```

## Train the paper preset

```bash
uv run python -m src.cli.train \
  --config configs/tben/dinov2_q_rois_all.yaml \
  --subject 1 \
  --output-root ./results
```

Artifacts are written under the run directory and include:

- `last.pth`
- `best.pth`
- `config.json`
- `metrics.json`

## Evaluate a saved run

```bash
uv run python -m src.cli.evaluate \
  --config configs/tben/dinov2_q_rois_all.yaml \
  --subject 1 \
  --checkpoint ./results/subj01_dinov2_q_rois_all/last.pth
```

## Export prediction shapes

```bash
uv run python -m src.cli.predict \
  --config configs/tben/dinov2_q_rois_all.yaml \
  --subject 1 \
  --checkpoint ./results/subj01_dinov2_q_rois_all/last.pth \
  --output ./results/subj01_dinov2_q_rois_all/prediction.json
```

## Wrapper usage

```python
import torch
from brain_encoder_wrapper import brain_encoder_wrapper

model = brain_encoder_wrapper(
    checkpoint_path="./results/subj01_dinov2_q_rois_all/last.pth",
    subject=1,
)
images = torch.randn(2, 3, 434, 434)
lh_pred, rh_pred = model.forward(images)
```

## Notebooks

- `run_model.ipynb`: launch CLI training commands from a notebook
- `test_wrapper.ipynb`: load a saved `.pth` artifact with `brain_encoder_wrapper`
- `visualize_results.ipynb`: inspect saved metrics and prediction metadata

## References

Adeli, H., Sun, M., & Kriegeskorte, N. (2025). Transformer brain encoders explain human high-level visual responses. arXiv preprint arXiv:2505.17329. [[arxiv](https://arxiv.org/abs/2505.17329)]
