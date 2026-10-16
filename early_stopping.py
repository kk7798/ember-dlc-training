"""Validation-loss early stopping adapter for DeepLabCut 3.0.0 pose training."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
from unittest.mock import patch


@dataclass
class LossMonitor:
    patience: int
    min_delta: float
    best_loss: float = math.inf
    reference_loss: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def __post_init__(self):
        if self.patience < 1:
            raise ValueError('PATIENCE must be at least 1.')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('MIN_DELTA must be finite and nonnegative.')

    def observe(self, epoch, loss):
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError(f'Non-finite validation loss at epoch {epoch}: {loss}')
        new_best = loss < self.best_loss
        if new_best:
            self.best_loss, self.best_epoch = loss, epoch
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return new_best, self.bad_epochs >= self.patience


class _StopTraining(Exception):
    pass


def fit_with_early_stopping(original_fit, runner, train_loader, valid_loader,
                            epochs, display_iters, patience, min_delta,
                            save_checkpoint, report_path):
    if len(valid_loader) == 0:
        raise ValueError('Early stopping requires nonempty validation data.')
    monitor = LossMonitor(patience, min_delta)
    manager = runner.snapshot_manager
    best_path = Path(manager.model_folder) / 'early-stopping' / 'best-validation.pt'
    best_path.parent.mkdir(parents=True, exist_ok=True)
    original_update = manager.update
    stopped_early = False
    checks = 0
    last_epoch = None
    previous_eval_count = len(runner.history['eval_loss'])

    def update(epoch, state_dict, last=False):
        nonlocal checks, last_epoch, previous_eval_count
        history = runner.history['eval_loss']
        if len(history) != previous_eval_count + 1:
            raise RuntimeError('Expected one fresh validation loss per epoch; DLC runner behavior changed.')
        previous_eval_count = len(history)
        new_best, stop = monitor.observe(epoch, history[-1])
        checks += 1
        last_epoch = epoch
        if new_best:
            temporary = best_path.with_suffix('.tmp')
            save_checkpoint(state_dict, temporary)
            temporary.replace(best_path)
        # Preserve DLC's regular snapshot policy, forcing a final save on stopping.
        original_update(epoch, state_dict, last=last or stop)
        print(f'[early stopping] epoch={epoch} validation_loss={float(history[-1]):.6g} '
              f'best={monitor.best_loss:.6g} patience={monitor.bad_epochs}/{patience}', flush=True)
        if stop:
            raise _StopTraining()

    # Patch only this runner instance; restore both attributes on failure or success.
    with patch.object(runner, 'eval_interval', 1), patch.object(manager, 'update', update):
        try:
            original_fit(runner, train_loader, valid_loader, epochs, display_iters)
        except _StopTraining:
            stopped_early = True
            print(f'[early stopping] Stopped at epoch {last_epoch}; '
                  f'best epoch={monitor.best_epoch}, checkpoint={best_path}', flush=True)
    if checks == 0:
        raise RuntimeError('Training finished without any validation checks.')
    report = {
        'enabled': True, 'monitor': 'validation_loss', 'mode': 'min',
        'patience': patience, 'min_delta': min_delta, 'validation_interval_epochs': 1,
        'stopped_early': stopped_early, 'epochs_completed': checks,
        'last_epoch': last_epoch, 'best_epoch': monitor.best_epoch,
        'best_validation_loss': monitor.best_loss,
        'best_checkpoint': str(best_path),
        'best_checkpoint_relative_to_project': str(best_path.relative_to(report_path.parent)),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'[early stopping] Best checkpoint saved at {best_path}', flush=True)


@contextmanager
def early_stopping(enabled, patience, min_delta, project_path):
    if not enabled:
        yield
        return
    import deeplabcut
    import torch
    from deeplabcut.pose_estimation_pytorch.runners.train import PoseTrainingRunner

    if deeplabcut.__version__ != '3.0.0':
        raise RuntimeError('This early-stopping adapter requires DeepLabCut 3.0.0.')
    LossMonitor(patience, min_delta)
    original_fit = PoseTrainingRunner.fit
    report_path = Path(project_path) / 'early_stopping.json'

    def fit(runner, train_loader, valid_loader, epochs, display_iters):
        return fit_with_early_stopping(
            original_fit, runner, train_loader, valid_loader, epochs, display_iters,
            patience, min_delta, torch.save, report_path,
        )

    with patch.object(PoseTrainingRunner, 'fit', fit):
        yield
