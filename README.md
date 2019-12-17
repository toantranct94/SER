# SER

Speech Emotion Recogniton

## Enviroment

Python 3

## Installation

```
pip install -r requirements.txt
```

## Data Preparing

Modify ```config.py``` and replace those following paths to your data paths

```
BASE_TRAIN = path_to_train_audio_file

BASE_PUBLIC_TEST = path_to_test_audio_file

TRAINING_GT = path_to_train_csv_file
```

## Traning

```
python train.py
```

Some best models will be saved to ```saved_model```

## Testing 
Open ```test.py``` and mofify ```model_path_fns ``` to models's path

For using singal model

```
model_path_fns  = [path-to-your-model]
```

For using multi models

```
model_path_fns  = [path-to-your-model-1, path-to-your-model-2, path-to-your-model-3,...]
```

Run

```
python test.py
```
