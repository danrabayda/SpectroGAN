import kagglehub
import os
import sys

def load_data(train_splits, test_split, batch_size, window_size_secs):
    data_folder = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")

    sys.path.insert(0, os.path.abspath(data_folder))
    from utils import ESC50

    shared_params = {
        'csv_path': data_folder+'/esc50.csv',
        'wav_dir': data_folder+'/audio/audio',
        'dest_dir': data_folder+'/audio/16000',
        'audio_rate': 16000,
        'only_ESC10': True,
        'pad': 0,
        'normalize': True
    }

    train_gen = ESC50(
        folds=train_splits,
        randomize=True,
        strongAugment=True,
        random_crop=True,
        inputLength=window_size_secs,
        mix=True,
        **shared_params
    ).batch_gen(batch_size)

    test_gen = ESC50(
        folds=[test_split],
        randomize=False,
        strongAugment=False,
        random_crop=True,
        inputLength=window_size_secs,
        mix=False,
        **shared_params
    ).batch_gen(batch_size)

    return train_gen, test_gen, data_folder