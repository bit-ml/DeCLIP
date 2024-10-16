def get_dolos_localisation_dataset_paths(dataset):
    paths = dict(
        fake_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    )
    return paths

def get_dolos_detection_dataset_paths(dataset):
    paths = dict(
        real_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        fake_path=f'datasets/dolos_data/celebahq/real/{dataset}/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    ),
    return paths

def get_autosplice_localisation_dataset_paths(compression):
    paths = dict(
        fake_path=f'datasets/AutoSplice/Forged_JPEG{compression}',
        masks_path=f'datasets/AutoSplice/Mask',
        key=f'autosplice_jpeg{compression}'
    )
    return paths

LOCALISATION_DATASET_PATHS = [
    get_dolos_localisation_dataset_paths('pluralistic'),
    get_dolos_localisation_dataset_paths('lama'),
    get_dolos_localisation_dataset_paths('repaint-p2-9k'),
    get_dolos_localisation_dataset_paths('ldm'),
    get_dolos_localisation_dataset_paths('ldm_clean'),
    get_dolos_localisation_dataset_paths('ldm_real'),

    get_autosplice_localisation_dataset_paths("75"),
    get_autosplice_localisation_dataset_paths("90"),
    get_autosplice_localisation_dataset_paths("100"),
]


DETECTION_DATASET_PATHS = [
    get_dolos_detection_dataset_paths('pluralistic'),
    get_dolos_detection_dataset_paths('lama'),
    get_dolos_detection_dataset_paths('repaint-p2-9k'),
    get_dolos_detection_dataset_paths('ldm'),
    get_dolos_detection_dataset_paths('ldm_clean'),
    get_dolos_detection_dataset_paths('ldm_real'),
]
