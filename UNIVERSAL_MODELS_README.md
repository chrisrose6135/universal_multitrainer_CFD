# Universal Direction Models Branch

This branch adds a pooled/universal-model workflow while keeping the existing symbol-specific training, replay, and live scripts available.

## What is new

New scripts/modules:

- `src/universal_symbol_features.py`
- `src/combine_universal_direction_datasets.py`
- `src/train_universal_direction_models.py`
- `src/replay_universal_direction_models.py`
- `src/live_universal_direction_policy.py`
- `config/direction_settings_universal_models.yaml`

New model architectures added to `src/direction_model.py`:

- `tsmixer` / `tsmixer_direction_v1`
- `patch_tst` / `patch_tst_direction_v1`
- `timesnet` / `timesnet_direction_v1`

The existing architecture grid also includes those three names in `config/direction_param_grid_model_architectures.yaml`.

## Why the universal combiner is group-aware

A universal dataset concatenates multiple symbol time series. If a normal sliding window were used blindly, sequences could cross from the end of one symbol into the start of another. The combiner therefore adds:

- `symbol`
- `sym_<SYMBOL>` one-hot columns
- `universal_sequence_group`
- `universal_split`

`prepare_direction_arrays()` uses `universal_sequence_group` only when `universal.enabled: true`, so symbol-specific training is unchanged.

`train_direction_policy.train_symbol()` uses `universal_split` only when `universal.enabled: true`, so universal train/validation splits are made per symbol rather than by concatenated row order.

## Workflow

### 1. Combine symbol datasets

```bash
python -m src.combine_universal_direction_datasets \
  --config config/direction_settings_universal_models.yaml \
  --force
```

Optional date range:

```bash
python -m src.combine_universal_direction_datasets \
  --config config/direction_settings_universal_models.yaml \
  --date-start 2020-01-01 \
  --date-end 2025-01-01 \
  --force
```

Output:

- `data/universal/UNIVERSAL_M5_direction_training.csv`
- `data/universal/UNIVERSAL_M5_direction_training.summary.json`

### 2. Train universal BUY/SELL models

Train all configured architectures:

```bash
python -m src.train_universal_direction_models \
  --config config/direction_settings_universal_models.yaml \
  --epochs 50 \
  --batch-size 256 \
  --device cuda
```

Train only the three new architectures:

```bash
python -m src.train_universal_direction_models \
  --config config/direction_settings_universal_models.yaml \
  --architectures tsmixer patch_tst timesnet \
  --sides buy sell \
  --epochs 50 \
  --batch-size 256 \
  --device cuda
```

Output:

- `models/universal/universal_model_manifest.json`
- `models/universal/universal_training_summary.json`
- one model/scaler/features bundle per architecture and side

### 3. Replay universal models per symbol

```bash
python -m src.replay_universal_direction_models \
  --config config/direction_settings_universal_models.yaml \
  --manifest models/universal/universal_model_manifest.json \
  --replay-start 2025-01-01 \
  --replay-end 2026-01-01 \
  --min-trades 50 \
  --min-net-pips 0 \
  --min-average-net-pips 0 \
  --device cuda
```

Output:

- `logs/universal_replay/universal_replay_by_symbol_model.csv`
- `logs/universal_replay/universal_live_models_enabled.csv`
- `logs/universal_replay/universal_live_models_disabled.csv`
- `logs/universal_replay/universal_live_ensemble_manifest.json`

### 4. Run universal live/demo/paper

```bash
python -m src.live_universal_direction_policy \
  --live-root "For Live Trading Universal" \
  --manifest logs/universal_replay/universal_live_ensemble_manifest.json \
  --universal-config config/direction_settings_universal_models.yaml \
  --mode paper \
  --data-source mt5 \
  --poll-seconds 20
```

Use `--mode demo` or `--mode live` only after paper validation.

## Existing symbol-specific pipeline

The symbol-specific pipeline remains available:

- `src/train_direction_policy_replay_each_epoch.py`
- `src/replay_direction_epochs.py`
- `src/select_live_trading_models.py`
- `src/live_direction_policy.py`
- `src/live_direction_policy_ensemble.py`

Universal symbol features and split handling are opt-in through:

```yaml
universal:
  enabled: true
```

Normal symbol-specific configs do not have this enabled, so their feature columns and split behaviour are not changed.
