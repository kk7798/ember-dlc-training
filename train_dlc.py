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

from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    merge_annotateddatasets,
)


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


def env_optional_float(name):
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return None

    return float(value)


# ============================================================
# PENNSIEVE-CONTROLLED PARAMETERS
# ============================================================

NET_TYPE = env_str(
    "NET_TYPE",
    "hrnet_w32"
)

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

SHUFFLE = env_int(
    "SHUFFLE",
    1
)

DEVICE = env_str(
    "DEVICE",
    "cuda:0"
)

SEED = env_int(
    "SEED",
    42
)

# Optional optimizer overrides
LEARNING_RATE = env_optional_float(
    "LEARNING_RATE"
)

WEIGHT_DECAY = env_optional_float(
    "WEIGHT_DECAY"
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
    print(message, flush=True)


# ============================================================
# VALIDATE PARAMETERS
# ============================================================

def validate_parameters():

    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be > 0.")

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0.")

    if SAVE_EPOCHS <= 0:
        raise ValueError("SAVE_EPOCHS must be > 0.")

    if MAX_SNAPSHOTS <= 0:
        raise ValueError("MAX_SNAPSHOTS must be > 0.")

    if DISPLAY_ITERS <= 0:
        raise ValueError("DISPLAY_ITERS must be > 0.")

    if SHUFFLE <= 0:
        raise ValueError("SHUFFLE must be >= 1.")

    models = list(available_models())

    if NET_TYPE not in models:
        raise ValueError(
            f"Unsupported DLC PyTorch model: {NET_TYPE}\n\n"
            f"Available models:\n"
            + "\n".join(models)
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# FIND CONFIG
# ============================================================

def find_config(root):

    configs = list(
        root.rglob("config.yaml")
    )

    if len(configs) == 0:
        raise FileNotFoundError(
            f"No config.yaml found under:\n{root}"
        )

    if len(configs) > 1:
        raise RuntimeError(
            "More than one config.yaml was found.\n"
            "This processor expects exactly one DLC project:\n\n"
            + "\n".join(str(x) for x in configs)
        )

    return configs[0]


# ============================================================
# VALIDATE LABELED DATA
# ============================================================

def validate_labeled_data(config_path):

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
            f"Missing labeled-data directory:\n{labeled_root}"
        )

    h5_files = sorted(
        labeled_root.rglob(
            f"CollectedData_{scorer}.h5"
        )
    )

    if not h5_files:
        raise RuntimeError(
            f"No CollectedData_{scorer}.h5 files found."
        )

    log("Annotation files:")

    for file in h5_files:
        log(f"  {file}")

    # --------------------------------------------------------
    # Protect EMBER test data from leakage
    # --------------------------------------------------------

    test_files = [
        f
        for f in h5_files
        if f.parent.name.lower().endswith("_test")
    ]

    if test_files:
        raise RuntimeError(
            "TEST DATA FOUND IN TRAINING PROJECT.\n\n"
            "The DLC training container must receive only "
            "EMBER train and val data.\n\n"
            + "\n".join(str(f) for f in test_files)
        )

    return cfg


# ============================================================
# CREATE THE FIXED EMBER TRAIN / VAL SPLIT
# ============================================================

def build_fixed_split(config_path):

    """
    Merge DLC annotation files using DLC's own merging logic,
    then assign indices according to the EMBER folder names:

        *_train -> training
        *_val   -> validation

    No random splitting occurs.
    """

    cfg = auxiliaryfunctions.read_config(
        str(config_path)
    )

    project_path = Path(
        cfg["project_path"]
    )

    training_set_folder = (
        project_path
        / auxiliaryfunctions.get_training_set_folder(cfg)
    )

    training_set_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # DLC merges in the same ordering it will use later when
    # creating the actual training dataset.
    data = merge_annotateddatasets(
        cfg,
        training_set_folder
    )

    if data is None:
        raise RuntimeError(
            "DeepLabCut could not merge the labeled datasets."
        )

    if len(data) == 0:
        raise RuntimeError(
            "Merged DLC annotation dataset is empty."
        )

    train_indices = []
    val_indices = []

    unknown_folders = set()

    for index, row_index in enumerate(data.index):

        # DLC image index should look like:
        #
        # ("labeled-data", "mice_hc_train", "img....png")
        #
        if isinstance(row_index, tuple):
            folder_name = str(row_index[1])
        else:
            parts = Path(str(row_index)).parts

            if len(parts) < 2:
                raise RuntimeError(
                    f"Unexpected DLC row index: {row_index}"
                )

            folder_name = parts[-2]

        folder_lower = folder_name.lower()

        if folder_lower.endswith("_train"):

            train_indices.append(index)

        elif folder_lower.endswith("_val"):

            val_indices.append(index)

        elif folder_lower.endswith("_test"):

            raise RuntimeError(
                "Test data entered the merged DLC dataset:\n"
                f"{folder_name}"
            )

        else:

            unknown_folders.add(folder_name)

    if unknown_folders:
        raise RuntimeError(
            "Could not determine train/val membership for these "
            "labeled-data folders:\n\n"
            + "\n".join(sorted(unknown_folders))
            + "\n\nFolder names must end in _train or _val."
        )

    if len(train_indices) == 0:
        raise RuntimeError(
            "No *_train samples were found."
        )

    if len(val_indices) == 0:
        raise RuntimeError(
            "No *_val samples were found."
        )

    total = (
        len(train_indices)
        + len(val_indices)
    )

    fixed_fraction = round(
        len(train_indices) / total,
        2
    )

    log()
    log("Fixed EMBER split:")
    log(
        f"  Training images:   {len(train_indices)}"
    )
    log(
        f"  Validation images: {len(val_indices)}"
    )
    log(
        f"  Total:             {total}"
    )
    log()
    log(
        "No random train/validation split will be created."
    )

    return (
        train_indices,
        val_indices,
        fixed_fraction,
    )


# ============================================================
# RUN METADATA
# ============================================================

def write_run_info(
    status,
    extra=None
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    info = {
        "status": status,
        "time": datetime.now().isoformat(),
        "deeplabcut_version": getattr(
            deeplabcut,
            "__version__",
            "unknown"
        ),
        "torch_version": torch.__version__,
        "net_type": NET_TYPE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "save_epochs": SAVE_EPOCHS,
        "max_snapshots": MAX_SNAPSHOTS,
        "display_iters": DISPLAY_ITERS,
        "shuffle": SHUFFLE,
        "device": DEVICE,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "split_policy": (
            "Predefined EMBER train/val split; "
            "EMBER test excluded"
        ),
    }

    if torch.cuda.is_available():

        info["gpu"] = (
            torch.cuda.get_device_name(0)
        )

        info["cuda_version"] = (
            torch.version.cuda
        )

    if extra:
        info.update(extra)

    (
        OUTPUT_DIR
        / "dlc_training_run.json"
    ).write_text(
        json.dumps(
            info,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# OPTIONAL PYTORCH OVERRIDES
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

    return updates or None


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

    validate_parameters()

    set_random_seeds()

    # --------------------------------------------------------
    # SOFTWARE / GPU
    # --------------------------------------------------------

    log(
        f"DeepLabCut version: "
        f"{getattr(deeplabcut, '__version__', 'unknown')}"
    )

    log(
        f"PyTorch version:    {torch.__version__}"
    )

    log(
        f"CUDA available:     {torch.cuda.is_available()}"
    )

    if DEVICE.startswith("cuda"):

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DEVICE={DEVICE}, but CUDA is unavailable."
            )

        log(
            f"GPU:                "
            f"{torch.cuda.get_device_name(0)}"
        )

        log(
            f"CUDA runtime:       {torch.version.cuda}"
        )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    log()
    log("Training parameters:")
    log(f"  NET_TYPE       = {NET_TYPE}")
    log(f"  EPOCHS         = {EPOCHS}")
    log(f"  BATCH_SIZE     = {BATCH_SIZE}")
    log(f"  SAVE_EPOCHS    = {SAVE_EPOCHS}")
    log(f"  MAX_SNAPSHOTS  = {MAX_SNAPSHOTS}")
    log(f"  DISPLAY_ITERS  = {DISPLAY_ITERS}")
    log(f"  SHUFFLE        = {SHUFFLE}")
    log(f"  DEVICE         = {DEVICE}")
    log(f"  SEED           = {SEED}")
    log(f"  LEARNING_RATE  = {LEARNING_RATE}")
    log(f"  WEIGHT_DECAY   = {WEIGHT_DECAY}")

    log()
    log(f"INPUT_DIR  = {INPUT_DIR}")
    log(f"OUTPUT_DIR = {OUTPUT_DIR}")

    # --------------------------------------------------------
    # FIND INPUT PROJECT
    # --------------------------------------------------------

    source_config = find_config(
        INPUT_DIR
    )

    source_project = (
        source_config.parent
    )

    log()
    log("Input DLC project:")
    log(str(source_project))

    # --------------------------------------------------------
    # COPY PROJECT TO WRITABLE OUTPUT LOCATION
    # --------------------------------------------------------

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

    log()
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

    # --------------------------------------------------------
    # UPDATE DLC CONFIG
    # --------------------------------------------------------

    cfg = auxiliaryfunctions.read_config(
        str(config_path)
    )

    cfg["project_path"] = str(
        output_project
    )

    cfg["engine"] = "pytorch"

    cfg["default_net_type"] = (
        NET_TYPE
    )

    auxiliaryfunctions.write_config(
        str(config_path),
        cfg
    )

    validate_labeled_data(
        config_path
    )

    # --------------------------------------------------------
    # BUILD PREDEFINED TRAIN / VAL INDICES
    # --------------------------------------------------------

    (
        train_indices,
        val_indices,
        fixed_fraction,
    ) = build_fixed_split(
        config_path
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DLC internally requires a TrainingFraction entry to
    # identify/name the training dataset.
    #
    # This value is DERIVED from the existing EMBER split.
    # It is NOT used to randomly split anything.
    # --------------------------------------------------------

    cfg = auxiliaryfunctions.read_config(
        str(config_path)
    )

    cfg["TrainingFraction"] = [
        fixed_fraction
    ]

    auxiliaryfunctions.write_config(
        str(config_path),
        cfg
    )

    log()
    log(
        f"DLC internal split metadata fraction: "
        f"{fixed_fraction:.2f}"
    )

    log(
        "(derived automatically; not a hyperparameter)"
    )

    # --------------------------------------------------------
    # SAVE RUN INFO
    # --------------------------------------------------------

    write_run_info(
        "starting",
        {
            "train_images":
                len(train_indices),
            "validation_images":
                len(val_indices),
            "derived_dlc_fraction":
                fixed_fraction,
        }
    )

    # --------------------------------------------------------
    # CREATE DLC TRAINING DATASET USING FIXED INDICES
    # --------------------------------------------------------

    log()
    log(
        "=============================================="
    )
    log(
        "Creating DLC dataset from fixed EMBER split"
    )
    log(
        "=============================================="
    )

    deeplabcut.create_training_dataset(
        str(config_path),

        Shuffles=[
            SHUFFLE
        ],

        # One explicitly frozen split
        trainIndices=[
            train_indices
        ],

        testIndices=[
            val_indices
        ],

        net_type=NET_TYPE,

        engine=Engine.PYTORCH,

        userfeedback=False,
    )

    log()
    log(
        "DLC training dataset created successfully."
    )

    # --------------------------------------------------------
    # OPTIONAL OPTIMIZER OVERRIDES
    # --------------------------------------------------------

    pytorch_updates = (
        build_pytorch_updates()
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    write_run_info(
        "training",
        {
            "train_images":
                len(train_indices),
            "validation_images":
                len(val_indices),
            "derived_dlc_fraction":
                fixed_fraction,
        }
    )

    log()
    log(
        "=============================================="
    )
    log(
        "Starting DeepLabCut training"
    )
    log(
        "=============================================="
    )

    deeplabcut.train_network(
        str(config_path),

        shuffle=SHUFFLE,

        # There is exactly one internally registered
        # fixed EMBER split.
        trainingsetindex=0,

        device=DEVICE,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        save_epochs=SAVE_EPOCHS,

        max_snapshots_to_keep=
            MAX_SNAPSHOTS,

        display_iters=DISPLAY_ITERS,

        pytorch_cfg_updates=
            pytorch_updates,
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    write_run_info(
        "completed",
        {
            "finished":
                datetime.now().isoformat(),
            "train_images":
                len(train_indices),
            "validation_images":
                len(val_indices),
            "derived_dlc_fraction":
                fixed_fraction,
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
        f"Output project: {output_project}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n==============================================",
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
                    "error": str(exc),
                    "failed":
                        datetime.now().isoformat(),
                }
            )

        except Exception:
            pass

        sys.exit(1)
