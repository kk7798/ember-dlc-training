import json
import os
import random
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import deeplabcut

from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch import available_models
from deeplabcut.utils import auxiliaryfunctions


# ============================================================
# ENVIRONMENT VARIABLE HELPERS
# ============================================================

def env_str(name, default):
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return default

    return value.strip()


def env_int(name, default):
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return default

    return int(value)


def env_float(name, default):
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return default

    return float(value)


def env_optional_int(name):
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return None

    return int(value)


def env_optional_float(name):
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return None

    return float(value)


def env_optional_str(name):
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return None

    return value.strip()


# ============================================================
# PENNSIEVE-CONTROLLED PARAMETERS
# ============================================================

# Network architecture
NET_TYPE = env_str(
    "NET_TYPE",
    "hrnet_w32"
)

# Pose training
EPOCHS = env_int(
    "EPOCHS",
    200
)

BATCH_SIZE = env_int(
    "BATCH_SIZE",
    2
)

SAVE_EPOCHS = env_int(
    "SAVE_EPOCHS",
    10
)

MAX_SNAPSHOTS = env_int(
    "MAX_SNAPSHOTS",
    5
)

DISPLAY_ITERS = env_int(
    "DISPLAY_ITERS",
    100
)

# DLC dataset / shuffle
SHUFFLE = env_int(
    "SHUFFLE",
    1
)

TRAININGSET_INDEX = env_int(
    "TRAININGSET_INDEX",
    0
)

TRAIN_FRACTION = env_float(
    "TRAIN_FRACTION",
    0.95
)

# Runtime
DEVICE = env_str(
    "DEVICE",
    "cuda:0"
)

SEED = env_int(
    "SEED",
    42
)

# Optional optimizer overrides
#
# Leave blank in Pennsieve to use DLC defaults.
LEARNING_RATE = env_optional_float(
    "LEARNING_RATE"
)

WEIGHT_DECAY = env_optional_float(
    "WEIGHT_DECAY"
)

# Optional top-down detector parameters
#
# These are ignored for normal bottom-up HRNet-W32 unless
# a detector-based architecture is selected.
DETECTOR_TYPE = env_optional_str(
    "DETECTOR_TYPE"
)

DETECTOR_BATCH_SIZE = env_optional_int(
    "DETECTOR_BATCH_SIZE"
)

DETECTOR_EPOCHS = env_optional_int(
    "DETECTOR_EPOCHS"
)

DETECTOR_SAVE_EPOCHS = env_optional_int(
    "DETECTOR_SAVE_EPOCHS"
)


# ============================================================
# PENNSIEVE DIRECTORIES
# ============================================================

INPUT_DIR = Path(
    os.environ.get(
        "INPUT_DIR",
        "/input"
    )
).resolve()

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        "/output"
    )
).resolve()


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(
        message,
        flush=True
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_parameters():

    if EPOCHS <= 0:
        raise ValueError(
            "EPOCHS must be > 0."
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be > 0."
        )

    if SAVE_EPOCHS <= 0:
        raise ValueError(
            "SAVE_EPOCHS must be > 0."
        )

    if MAX_SNAPSHOTS <= 0:
        raise ValueError(
            "MAX_SNAPSHOTS must be > 0."
        )

    if SHUFFLE <= 0:
        raise ValueError(
            "SHUFFLE must be >= 1."
        )

    if TRAININGSET_INDEX < 0:
        raise ValueError(
            "TRAININGSET_INDEX must be >= 0."
        )

    if not 0 < TRAIN_FRACTION < 1:
        raise ValueError(
            "TRAIN_FRACTION must be between 0 and 1."
        )

    models = list(
        available_models()
    )

    if NET_TYPE not in models:

        raise ValueError(
            f"Unsupported PyTorch DLC model: {NET_TYPE}\n\n"
            f"Available models:\n"
            + "\n".join(models)
        )


# ============================================================
# RANDOM SEEDS
# ============================================================

def set_random_seeds():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


# ============================================================
# FIND DLC PROJECT
# ============================================================

def find_config(root):

    configs = list(
        root.rglob(
            "config.yaml"
        )
    )

    if len(configs) == 0:

        raise FileNotFoundError(
            "No DeepLabCut config.yaml found under:\n"
            f"{root}"
        )

    if len(configs) > 1:

        raise RuntimeError(
            "More than one config.yaml was found.\n"
            "This processor expects exactly one DLC project.\n\n"
            + "\n".join(
                str(x)
                for x in configs
            )
        )

    return configs[0]


# ============================================================
# VERIFY PROJECT
# ============================================================

def validate_project(
    config_path
):

    cfg = auxiliaryfunctions.read_config(
        str(config_path)
    )

    project_path = Path(
        cfg["project_path"]
    )

    scorer = cfg["scorer"]

    labeled_root = (
        project_path
        / "labeled-data"
    )

    if not labeled_root.is_dir():

        raise RuntimeError(
            "No labeled-data directory found:\n"
            f"{labeled_root}"
        )

    annotation_files = list(
        labeled_root.rglob(
            f"CollectedData_{scorer}.h5"
        )
    )

    if len(annotation_files) == 0:

        raise RuntimeError(
            "No DLC annotation H5 files found under:\n"
            f"{labeled_root}\n\n"
            f"Expected files named "
            f"CollectedData_{scorer}.h5"
        )

    log(
        f"Found {len(annotation_files)} "
        "DLC annotation file(s):"
    )

    for f in annotation_files:

        log(
            f"  {f}"
        )

    return cfg


# ============================================================
# SAVE RUN INFORMATION
# ============================================================

def write_run_info(
    status,
    extra=None
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    information = {
        "status": status,
        "time": datetime.now().isoformat(),
        "deeplabcut_version":
            getattr(
                deeplabcut,
                "__version__",
                "unknown"
            ),
        "torch_version":
            torch.__version__,
        "network":
            NET_TYPE,
        "epochs":
            EPOCHS,
        "batch_size":
            BATCH_SIZE,
        "save_epochs":
            SAVE_EPOCHS,
        "max_snapshots":
            MAX_SNAPSHOTS,
        "display_iters":
            DISPLAY_ITERS,
        "shuffle":
            SHUFFLE,
        "trainingset_index":
            TRAININGSET_INDEX,
        "training_fraction":
            TRAIN_FRACTION,
        "device":
            DEVICE,
        "seed":
            SEED,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "detector_type":
            DETECTOR_TYPE,
        "detector_batch_size":
            DETECTOR_BATCH_SIZE,
        "detector_epochs":
            DETECTOR_EPOCHS,
        "detector_save_epochs":
            DETECTOR_SAVE_EPOCHS,
    }

    if torch.cuda.is_available():

        information[
            "gpu"
        ] = torch.cuda.get_device_name(0)

        information[
            "cuda_version"
        ] = torch.version.cuda

    if extra is not None:

        information.update(
            extra
        )

    run_file = (
        OUTPUT_DIR
        / "dlc_training_run.json"
    )

    run_file.write_text(
        json.dumps(
            information,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# BUILD PYTORCH CONFIG OVERRIDES
# ============================================================

def build_pytorch_updates():

    updates = {}

    if LEARNING_RATE is not None:

        updates[
            "runner.optimizer.params.lr"
        ] = LEARNING_RATE

    if WEIGHT_DECAY is not None:

        updates[
            "runner.optimizer.params.weight_decay"
        ] = WEIGHT_DECAY

    if len(updates) == 0:
        return None

    return updates


# ============================================================
# MAIN
# ============================================================

def main():

    log(
        "=============================================="
    )

    log(
        "EMBER DeepLabCut Training Processor"
    )

    log(
        "=============================================="
    )

    log()

    # ========================================================
    # VALIDATE PARAMETERS
    # ========================================================

    validate_parameters()

    set_random_seeds()

    # ========================================================
    # SOFTWARE INFORMATION
    # ========================================================

    log(
        f"DeepLabCut version: "
        f"{getattr(deeplabcut, '__version__', 'unknown')}"
    )

    log(
        f"PyTorch version:    "
        f"{torch.__version__}"
    )

    log(
        f"CUDA available:     "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        log(
            f"GPU:                "
            f"{torch.cuda.get_device_name(0)}"
        )

        log(
            f"CUDA runtime:       "
            f"{torch.version.cuda}"
        )

    log()

    # ========================================================
    # GPU CHECK
    # ========================================================

    if DEVICE.lower().startswith(
        "cuda"
    ):

        if not torch.cuda.is_available():

            raise RuntimeError(
                f"DEVICE={DEVICE}, but CUDA is not "
                "available inside the container."
            )

    # ========================================================
    # PRINT PARAMETERS
    # ========================================================

    log(
        "Pennsieve training parameters:"
    )

    log(
        f"  NET_TYPE            = {NET_TYPE}"
    )

    log(
        f"  EPOCHS              = {EPOCHS}"
    )

    log(
        f"  BATCH_SIZE          = {BATCH_SIZE}"
    )

    log(
        f"  SAVE_EPOCHS         = {SAVE_EPOCHS}"
    )

    log(
        f"  MAX_SNAPSHOTS       = {MAX_SNAPSHOTS}"
    )

    log(
        f"  DISPLAY_ITERS       = {DISPLAY_ITERS}"
    )

    log(
        f"  SHUFFLE             = {SHUFFLE}"
    )

    log(
        f"  TRAININGSET_INDEX   = {TRAININGSET_INDEX}"
    )

    log(
        f"  TRAIN_FRACTION      = {TRAIN_FRACTION}"
    )

    log(
        f"  DEVICE              = {DEVICE}"
    )

    log(
        f"  SEED                = {SEED}"
    )

    log(
        f"  LEARNING_RATE       = {LEARNING_RATE}"
    )

    log(
        f"  WEIGHT_DECAY        = {WEIGHT_DECAY}"
    )

    log(
        f"  DETECTOR_TYPE       = {DETECTOR_TYPE}"
    )

    log(
        f"  DETECTOR_BATCH_SIZE = "
        f"{DETECTOR_BATCH_SIZE}"
    )

    log(
        f"  DETECTOR_EPOCHS     = "
        f"{DETECTOR_EPOCHS}"
    )

    log(
        f"  DETECTOR_SAVE_EPOCHS = "
        f"{DETECTOR_SAVE_EPOCHS}"
    )

    log()

    log(
        f"INPUT_DIR  = {INPUT_DIR}"
    )

    log(
        f"OUTPUT_DIR = {OUTPUT_DIR}"
    )

    log()

    # ========================================================
    # LOCATE DLC PROJECT
    # ========================================================

    source_config = find_config(
        INPUT_DIR
    )

    source_project = (
        source_config.parent
    )

    log(
        "Input DLC project:"
    )

    log(
        str(source_project)
    )

    log()

    # ========================================================
    # COPY PROJECT INTO OUTPUT
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_project = (
        OUTPUT_DIR
        / "dlc_project"
    )

    if output_project.exists():

        shutil.rmtree(
            output_project
        )

    log(
        "Copying DLC project to OUTPUT_DIR..."
    )

    shutil.copytree(
        source_project,
        output_project
    )

    config_path = (
        output_project
        / "config.yaml"
    )

    # ========================================================
    # UPDATE DLC CONFIG
    # ========================================================

    cfg = auxiliaryfunctions.read_config(
        str(config_path)
    )

    # DLC must know the actual container path.
    cfg[
        "project_path"
    ] = str(
        output_project
    )

    # Force PyTorch DLC.
    cfg[
        "engine"
    ] = "pytorch"

    # Set current architecture.
    cfg[
        "default_net_type"
    ] = NET_TYPE

    # Control DLC's internal train/test split.
    cfg[
        "TrainingFraction"
    ] = [
        TRAIN_FRACTION
    ]

    auxiliaryfunctions.write_config(
        str(config_path),
        cfg
    )

    cfg = validate_project(
        config_path
    )

    log()

    log(
        f"Multi-animal project: "
        f"{cfg.get('multianimalproject', False)}"
    )

    if cfg.get(
        "multianimalproject",
        False
    ):

        log(
            f"Individuals: "
            f"{cfg.get('individuals', [])}"
        )

        log(
            f"Multi-animal bodyparts: "
            f"{cfg.get('multianimalbodyparts', [])}"
        )

        log(
            f"Identity learning: "
            f"{cfg.get('identity', False)}"
        )

    else:

        log(
            f"Bodyparts: "
            f"{cfg.get('bodyparts', [])}"
        )

    log()

    # ========================================================
    # RECORD RUN
    # ========================================================

    write_run_info(
        "starting"
    )

    # ========================================================
    # CREATE TRAINING DATASET
    # ========================================================

    log(
        "=============================================="
    )

    log(
        "Creating DeepLabCut PyTorch training dataset"
    )

    log(
        "=============================================="
    )

    create_kwargs = {
        "config":
            str(config_path),

        "Shuffles":
            [SHUFFLE],

        "net_type":
            NET_TYPE,

        "engine":
            Engine.PYTORCH,

        "userfeedback":
            False,
    }

    # Detector selection matters only for detector-based
    # / top-down models.
    if DETECTOR_TYPE is not None:

        create_kwargs[
            "detector_type"
        ] = DETECTOR_TYPE

    deeplabcut.create_training_dataset(
        **create_kwargs
    )

    log()

    log(
        "Training dataset created successfully."
    )

    log()

    # ========================================================
    # OPTIONAL PYTORCH CONFIG OVERRIDES
    # ========================================================

    pytorch_updates = (
        build_pytorch_updates()
    )

    if pytorch_updates is not None:

        log(
            "PyTorch configuration overrides:"
        )

        for key, value in (
            pytorch_updates.items()
        ):

            log(
                f"  {key} = {value}"
            )

        log()

    # ========================================================
    # TRAIN POSE MODEL
    # ========================================================

    log(
        "=============================================="
    )

    log(
        "Starting DeepLabCut training"
    )

    log(
        "=============================================="
    )

    write_run_info(
        "training"
    )

    train_kwargs = {
        "config":
            str(config_path),

        "shuffle":
            SHUFFLE,

        "trainingsetindex":
            TRAININGSET_INDEX,

        "device":
            DEVICE,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "save_epochs":
            SAVE_EPOCHS,

        "max_snapshots_to_keep":
            MAX_SNAPSHOTS,

        "displayiters":
            DISPLAY_ITERS,
    }

    # Optional detector overrides for top-down architectures.
    if DETECTOR_BATCH_SIZE is not None:

        train_kwargs[
            "detector_batch_size"
        ] = DETECTOR_BATCH_SIZE

    if DETECTOR_EPOCHS is not None:

        train_kwargs[
            "detector_epochs"
        ] = DETECTOR_EPOCHS

    if DETECTOR_SAVE_EPOCHS is not None:

        train_kwargs[
            "detector_save_epochs"
        ] = DETECTOR_SAVE_EPOCHS

    if pytorch_updates is not None:

        train_kwargs[
            "pytorch_cfg_updates"
        ] = pytorch_updates

    deeplabcut.train_network(
        **train_kwargs
    )

    # ========================================================
    # FINISH
    # ========================================================

    write_run_info(
        "completed",
        {
            "finished":
                datetime.now().isoformat(),
            "output_project":
                str(output_project),
        }
    )

    log()

    log(
        "=============================================="
    )

    log(
        "DEEPLABCUT TRAINING COMPLETE"
    )

    log(
        "=============================================="
    )

    log()

    log(
        "Output DLC project:"
    )

    log(
        str(output_project)
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n"
            "==============================================",
            file=sys.stderr
        )

        print(
            "DEEPLABCUT TRAINING FAILED",
            file=sys.stderr
        )

        print(
            "==============================================\n",
            file=sys.stderr
        )

        traceback.print_exc()

        try:

            write_run_info(
                "failed",
                {
                    "error":
                        str(exc),
                    "failed":
                        datetime.now().isoformat(),
                }
            )

        except Exception:

            pass

        sys.exit(1)
