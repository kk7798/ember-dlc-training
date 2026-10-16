# EMBER DeepLabCut training with early stopping

Based on the supplied v1.1.0 release, retaining its fixed train/validation split,
filesystem protections, and the later startup diagnostics.

## Parameters

- EARLY_STOPPING: true (default) or false. Enabled mode monitors pose-model
  validation loss once per epoch. Disabled mode leaves DLC training unchanged.
- PATIENCE: 10 by default. Stop after this many consecutive validation checks
  without a decrease greater than MIN_DELTA. Must be at least 1.
- MIN_DELTA: 0.0001 by default. Absolute loss decrease required to reset patience.
  Small cumulative improvements are compared with the last significant improvement.
- EPOCHS: still the maximum number of epochs.

The best checkpoint always tracks the actual lowest finite validation loss,
including improvements smaller than MIN_DELTA. It is saved under the DLC model's
training folder at early-stopping/best-validation.pt. This is a separate checkpoint,
not deleted by MAX_SNAPSHOTS, and is NOT automatically selected by ordinary DLC
snapshot selection. Select it explicitly when using the best model for inference
or resumed training. Standard DLC checkpoints continue to be saved as configured,
and a final snapshot update is forced when patience is exhausted.

The project root contains early_stopping.json with the best epoch, best loss,
completed epochs, stopped_early flag, and absolute and project-relative checkpoint
paths. The normal run metadata also records the early-stopping parameters. An
early stop completes successfully so downstream upload can proceed. Non-finite
validation losses and missing validation data fail explicitly.

## Compatibility

This adapter is deliberately restricted to DeepLabCut 3.0.0. It temporarily wraps
PoseTrainingRunner.fit and the runner's snapshot update, while retaining DLC's
training loop, optimizer, scheduler, loss calculations, and fixed split. Attributes
are restored on completion or failure. Validation runs every epoch when enabled,
which can add runtime compared with a larger validation interval. Object-detector
training, if a future model requires it, is not early-stopped by this adapter.
No patience setting is passed as an unsupported DLC configuration option.

## Deployment

Upload Dockerfile, entrypoint.sh, train_dlc.py, processor_utils.py,
early_stopping.py, app.yml, .dockerignore, .gitignore, and this README.md to the
ember-dlc-training repository root. Publish a new unused release tag. No DANDI
repository change is required.

For the one-epoch smoke test keep EPOCHS=1, EARLY_STOPPING=true, PATIENCE=10,
MIN_DELTA=0.0001. It should complete the epoch and preserve its best checkpoint;
there are not enough epochs for patience-based stopping.

## Verification

Five local unit tests passed. The original DeepLabCut v3.0.0 TrainingRunner.fit
source was also exercised with synthetic losses: loss sequence 1, 0.8, 0.9, 1
and patience 2 stopped at epoch 4 and preserved the epoch-2 checkpoint. A one-epoch
run completed normally. These are control-flow tests, not a live GPU training run.
The optional test_early_stopping.py is in the source folder, excluded from the
release ZIP, and can run with python -m unittest -v test_early_stopping.py.
