"""DeepLabCut 3.0 PyTorch inference and held-out EMBER evaluation."""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from processor_utils import ensure_separate_output, prepare_output

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.m4v'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def configure_caches(output):
    for variable, suffix in {'XDG_CACHE_HOME': '', 'TORCH_HOME': 'torch',
                             'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
                             'MPLCONFIGDIR': 'matplotlib'}.items():
        directory = output / '.cache' / suffix
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ.setdefault('MPLBACKEND', 'Agg')


def collect_files(location, extensions):
    location = Path(location).expanduser().resolve()
    files = [location] if location.is_file() else sorted(location.rglob('*'))
    files = [p for p in files if p.is_file() and p.suffix.lower() in extensions]
    if not files:
        raise ValueError(f'No supported files found in {location}')
    return files


def discover_project(root):
    """Find one trained project without relying on Pennsieve folder names."""
    root = Path(root)
    configs = sorted(p for p in root.rglob('config.yaml')
                     if not any(part.startswith('.') for part in p.relative_to(root).parts)
                     and (p.parent / 'dlc-models-pytorch').is_dir()
                     and any((p.parent / 'dlc-models-pytorch').rglob('snapshot*.pt')))
    if len(configs) != 1:
        raise ValueError(f'Expected one trained DLC project under {root}, found {len(configs)}; '
                         'supply --project or MODEL_DIR to choose one')
    return str(configs[0].parent.resolve())


def discover_data(location, project, cfg):
    """Classify user media, excluding model assets and training/validation data."""
    location, project = Path(location).resolve(), Path(project).resolve()
    if not location.exists():
        raise FileNotFoundError(location)
    if location.name.lower().endswith(('_train', '_val')):
        raise ValueError('Automatic inference does not accept training/validation folders')
    def allowed(path):
        relative = path.relative_to(location) if location.is_dir() else Path(path.name)
        if any(part.startswith('.') for part in relative.parts):
            return False
        if project != location and (path == project or project in path.parents):
            return False
        for part in relative.parts[:-1]:
            lower = part.lower()
            if (lower.endswith(('_train', '_val'))
                    or lower == 'training-datasets'
                    or (lower == 'labeled-data' and location == project)
                    or lower.startswith(('dlc-models', 'evaluation-results'))):
                return False
        return True
    files = sorted(p for p in (location.rglob('*') if location.is_dir() else [location])
                   if p.is_file() and allowed(p))
    test_dirs = set()
    for path in files:
        for parent in path.parents:
            if parent.name.lower().endswith('_test') and (parent == location or location in parent.parents):
                test_dirs.add(parent)
                break
    expected = f"{cfg['Task']}_test"
    for folder in test_dirs:
        if folder.name != expected:
            raise ValueError(f'Test folder {folder.name} does not match model task {cfg["Task"]}')
        if not (folder / f"CollectedData_{cfg['scorer']}.h5").is_file():
            raise ValueError(f'Missing ground-truth annotations in {folder}; supply labeled EMBER test data')
    if len(test_dirs) > 1:
        raise ValueError('Multiple EMBER test folders found; submit one test set per run')
    test_dir = next(iter(test_dirs), None)
    media = [p for p in files if test_dir is None or test_dir not in p.parents]
    images = [p for p in media if p.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [p for p in media if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not test_dir and not images and not videos:
        raise ValueError('No test images, external images or videos found outside the trained project')
    return str(test_dir) if test_dir else None, images, videos


def prepare_project(source, output):
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.name == 'config.yaml':
        source = source.parent
    if not (source / 'config.yaml').is_file():
        raise FileNotFoundError(f'Missing config.yaml in {source}')
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the source project')
    target = output / 'model'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        '.cache', '.git', '__pycache__', 'labeled-data', 'videos'))
    config_path = target / 'config.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get('engine') != 'pytorch':
        raise ValueError('A PyTorch project is required')
    cfg['project_path'] = str(target)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for path in target.rglob('pytorch_config.yaml'):
        model_cfg = yaml.safe_load(path.read_text())
        metadata = model_cfg.setdefault('metadata', {})
        metadata.update(project_path=str(target), pose_config_path=str(path))
        path.write_text(yaml.safe_dump(model_cfg, sort_keys=False))
    if not list(target.rglob('snapshot*.pt')):
        raise FileNotFoundError('No snapshot checkpoints found')
    return config_path, cfg


def annotation_images(folder, cfg):
    folder = Path(folder).expanduser().resolve()
    expected = f"{cfg['Task']}_test"
    if folder.name != expected:
        matches = sorted(p for p in folder.rglob(expected) if p.is_dir())
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {expected} folder, found {len(matches)}')
        folder = matches[0]
    labels = pd.read_hdf(folder / f"CollectedData_{cfg['scorer']}.h5", key='df_with_missing')
    if labels.empty or not labels.index.is_unique:
        raise ValueError('Empty or duplicate annotation rows')
    images = []
    for index in labels.index:
        parts = list(index) if isinstance(index, tuple) else str(index).replace('\\', '/').split('/')
        if len(parts) < 2 or str(parts[-2]) != folder.name:
            raise ValueError(f'Unexpected annotation index: {index}')
        name = str(parts[-1])
        if Path(name).name != name:
            raise ValueError(f'Invalid filename: {name}')
        image = folder / name
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Missing or unsupported image: {image}')
        images.append(image)
    if len(images) != len(set(images)):
        raise ValueError('Repeated annotation image paths')
    return labels, images


def table_to_array(table, bodyparts, prediction=False):
    if not {'scorer', 'bodyparts', 'coords'}.issubset(table.columns.names):
        raise ValueError('Unexpected DLC table column structure')
    scorers = table.columns.get_level_values('scorer').unique()
    if len(scorers) != 1:
        raise ValueError('Expected one scorer')
    table = table.xs(scorers[0], axis=1, level='scorer')
    if set(table.columns.get_level_values('bodyparts')) != set(bodyparts):
        raise ValueError('Model and table body parts differ')
    multi = 'individuals' in table.columns.names
    animals = list(table.columns.get_level_values('individuals').unique()) if multi else [None]
    coords = ['x', 'y', 'likelihood'] if prediction else ['x', 'y']
    result = np.full((len(table), len(animals), len(bodyparts), len(coords)), np.nan)
    for a, animal in enumerate(animals):
        subset = table.xs(animal, axis=1, level='individuals') if multi else table
        for b, bodypart in enumerate(bodyparts):
            for c, coord in enumerate(coords):
                result[:, a, b, c] = subset[(bodypart, coord)].to_numpy(dtype=float)
    return result


def evaluate_test(labels, images, prediction_file, cfg, cutoff, pck_pixels, output):
    predictions = pd.read_hdf(prediction_file, key='predictions')
    names = [str(i[-1] if isinstance(i, tuple) else i).replace('\\', '/').split('/')[-1]
             for i in predictions.index]
    expected = [p.name for p in images]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError('Prediction image rows do not match test annotations')
    predictions.index = names
    predictions = predictions.loc[expected]
    bodyparts = cfg['multianimalbodyparts'] if cfg.get('multianimalproject') else cfg['bodyparts']
    if cfg.get('uniquebodyparts'):
        raise ValueError('Unique-bodypart evaluation is not supported')
    truth = table_to_array(labels, bodyparts)
    predicted = table_to_array(predictions, bodyparts, prediction=True)
    records = []
    extra_animals = 0
    for f, image in enumerate(images):
        gt, pred = truth[f], predicted[f]
        visible = np.isfinite(gt).all(axis=-1)
        usable = np.isfinite(pred).all(axis=-1) & (pred[..., 2] >= cutoff)
        gs, ps = np.flatnonzero(visible.any(axis=1)), np.flatnonzero(usable.any(axis=1))
        assignments = {}
        if len(gs) and len(ps):
            distances = np.linalg.norm(gt[gs, None, :, :] - pred[None, ps, :, :2], axis=-1)
            valid = visible[gs, None, :] & usable[None, ps, :] & np.isfinite(distances)
            finite = distances[valid]
            penalty = max(pck_pixels + 1, float(finite.max()) + 1 if finite.size else 1)
            costs = np.where(valid, distances, penalty)
            costs = (costs * visible[gs, None, :]).sum(-1) / visible[gs].sum(-1)[:, None]
            rows, cols = linear_sum_assignment(costs)
            assignments = {int(gs[r]): int(ps[c]) for r, c in zip(rows, cols) if valid[r, c].any()}
        extra_animals += len(ps) - len(assignments)
        for animal in gs:
            slot = assignments.get(int(animal))
            for part in np.flatnonzero(visible[animal]):
                error = None
                if slot is not None and usable[slot, part]:
                    error = float(np.linalg.norm(gt[animal, part] - pred[slot, part, :2]))
                records.append(dict(image=image.name, ground_truth_slot=int(animal),
                                    prediction_slot=slot, bodypart=bodyparts[part], error_px=error,
                                    predicted=error is not None,
                                    correct=error is not None and error <= pck_pixels))
    if not records:
        raise ValueError('No finite labeled keypoints')
    details = pd.DataFrame(records)
    details.to_csv(output / 'keypoint_errors.csv', index=False)
    def summarize(frame):
        errors = frame.error_px.dropna().to_numpy(dtype=float)
        return dict(labeled_keypoints=len(frame), predicted_keypoints=int(frame.predicted.sum()),
                    coverage=float(frame.predicted.mean()), pck=float(frame.correct.mean()),
                    mean_error_px=float(errors.mean()) if errors.size else None,
                    rmse_px=float(np.sqrt(np.mean(errors ** 2))) if errors.size else None)
    report = dict(images=len(images), confidence_cutoff=cutoff, pck_threshold_px=pck_pixels,
                  matching='Per-frame Hungarian assignment; missing keypoints penalized',
                  error_denominator='Matched finite keypoints above confidence cutoff',
                  pck_denominator='All finite ground-truth keypoints; missing predictions fail',
                  unmatched_predicted_animals=extra_animals, overall=summarize(details),
                  per_bodypart={part: summarize(group) for part, group in details.groupby('bodypart', sort=False)})
    write_json(output / 'evaluation.json', report)
    return report


def run(args, output, info):
    import torch
    import deeplabcut
    from deeplabcut.pose_estimation_pytorch import analyze_images, analyze_videos
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    config, cfg = prepare_project(args.project, output)
    info.update(deeplabcut_version=deeplabcut.__version__, torch_version=torch.__version__, task=cfg['Task'])
    info['checkpoints'] = [str(p.relative_to(config.parent)) for p in sorted(config.parent.rglob('snapshot*.pt'))]
    info['training_fraction'] = cfg['TrainingFraction'][args.trainingsetindex]
    shared = dict(config=str(config), shuffle=args.shuffle, trainingsetindex=args.trainingsetindex,
                  snapshot_index=args.snapshot_index, detector_snapshot_index=args.detector_snapshot_index,
                  device=args.device, save_as_csv=True)
    image_files = collect_files(args.images, IMAGE_EXTENSIONS) if args.images else []
    video_files = collect_files(args.videos, VIDEO_EXTENSIONS) if args.videos else []
    test_folder = args.ember_test
    if args.data:
        test_folder, image_files, video_files = discover_data(args.data, args.project, cfg)
    info['detected_inputs'] = dict(ember_test=test_folder, images=[str(p) for p in image_files],
                                   videos=[str(p) for p in video_files])
    if test_folder:
        if cfg.get('uniquebodyparts'):
            raise ValueError('Unique-bodypart evaluation is not supported')
        labels, images = annotation_images(test_folder, cfg)
        destination = output / 'ember_test'
        destination.mkdir()
        analyze_images(**shared, images=[str(p) for p in images], output_dir=str(destination),
                       plotting=args.plot_images, pcutoff=args.pcutoff)
        files = list(destination.glob('image_predictions_*.h5'))
        if len(files) != 1:
            raise RuntimeError('Expected one image prediction HDF5')
        report = evaluate_test(labels, images, files[0], cfg, args.pcutoff, args.pck_pixels, destination)
        info['ember_test'] = report['overall']
        print(json.dumps(report['overall'], indent=2), flush=True)
    if image_files:
        groups = {}
        for image in image_files:
            groups.setdefault(image.parent, []).append(image)
        info['external_images'] = []
        for index, (parent, group) in enumerate(groups.items()):
            destination = output / 'images' / f'{index:04d}'
            destination.mkdir(parents=True)
            analyze_images(**shared, images=[str(p) for p in group], output_dir=str(destination),
                           plotting=args.plot_images, pcutoff=args.pcutoff)
            if len(list(destination.glob('image_predictions_*.h5'))) != 1:
                raise RuntimeError(f'No image prediction table produced for {parent}')
            info['external_images'].append(dict(source_directory=str(parent), files=[str(p) for p in group],
                                                output=str(destination.relative_to(output))))
    if video_files:
        info['videos'] = []
        for index, video in enumerate(video_files):
            destination = output / 'videos' / f'{index:04d}_{video.stem}'
            destination.mkdir(parents=True)
            analyze_videos(**shared, videos=[str(video)], destfolder=str(destination),
                           batch_size=args.batch_size, auto_track=False, save_as_df=True)
            files = list(destination.glob('*.h5'))
            if not files:
                raise RuntimeError(f'No prediction table produced for {video}')
            for path in files:
                pd.read_hdf(path).to_csv(path.with_suffix('.csv'))
            info['videos'].append(dict(source=str(video), output=str(destination.relative_to(output)),
                                       identity='Frame-local slots; no temporal tracking'))


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('true', '1', 'yes'):
        return True
    if value in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true/false, yes/no, or 1/0')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=os.environ.get('MODEL_DIR'))
    parser.add_argument('--output', default=os.environ.get('OUTPUT_DIR', '/output'))
    parser.add_argument('--data', default=os.environ.get('DATA_DIR'))
    for name in ['ember-test', 'images', 'videos']:
        parser.add_argument('--' + name)
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'cpu'))
    for name, env, default in [
        ('shuffle', 'SHUFFLE', 1), ('trainingsetindex', 'TRAININGSET_INDEX', 0),
        ('snapshot-index', 'SNAPSHOT_INDEX', -1),
        ('detector-snapshot-index', 'DETECTOR_SNAPSHOT_INDEX', -1),
        ('batch-size', 'BATCH_SIZE', 1), ('cpu-threads', 'CPU_THREADS', 4),
    ]:
        parser.add_argument('--' + name, type=int, default=os.environ.get(env, str(default)))
    parser.add_argument('--pcutoff', type=float, default=os.environ.get('PCUTOFF', '0.6'))
    parser.add_argument('--pck-pixels', type=float, default=os.environ.get('PCK_PIXELS', '10'))
    parser.add_argument('--plot-images', type=parse_bool, nargs='?', const=True,
                        default=os.environ.get('PLOT_IMAGES', 'false'))
    args = parser.parse_args(argv)
    root = Path(os.environ.get('INPUT_DIR', '/input')).expanduser().resolve()
    direct = any([args.ember_test, args.images, args.videos])
    if direct and args.data:
        parser.error('Use --data OR explicit --ember-test/--images/--videos, not both')
    args.input_root = str(root)
    if not direct:
        args.data = args.data or str(root)
    def input_path(value):
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else root / path).resolve())
    args.project = input_path(args.project) if args.project else None
    if args.project and Path(args.project).name == 'config.yaml':
        args.project = str(Path(args.project).parent)
    for name in ['data', 'ember_test', 'images', 'videos']:
        if getattr(args, name):
            setattr(args, name, input_path(getattr(args, name)))
    if not 0 < args.pcutoff < 1 or not np.isfinite(args.pck_pixels) or args.pck_pixels <= 0:
        parser.error('Invalid confidence cutoff or PCK threshold')
    if min(args.batch_size, args.shuffle, args.cpu_threads) < 1 or args.trainingsetindex < 0:
        parser.error('Invalid batch size, shuffle, CPU threads or training-set index')
    if min(args.snapshot_index, args.detector_snapshot_index) < -1:
        parser.error('Snapshot indices must be -1 (latest) or nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if not args.project:
        args.project = discover_project(args.input_root)
    ensure_separate_output(output, args.project, args.data, args.ember_test, args.images, args.videos)
    prepare_output(output)
    info = dict(status='starting', started=datetime.now(timezone.utc).isoformat(), parameters=vars(args))
    try:
        # Fail on missing inputs before importing DLC or copying model weights.
        if not Path(args.project).exists():
            raise FileNotFoundError(args.project)
        for source in [args.data, args.ember_test, args.images, args.videos]:
            if source and not Path(source).exists():
                raise FileNotFoundError(source)
        configure_caches(output)
        write_json(output / 'inference_run.json', info)
        run(args, output, info)
        info['status'] = 'completed'
    except Exception as exc:
        info.update(status='failed', error=str(exc))
        raise
    finally:
        info['finished'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'inference_run.json', info)


if __name__ == '__main__':
    main()
