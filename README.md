# Forza autopilot

AI autopilot for Forza Horizon 4

## What it does?

This autopilot drives as you learn it to drive, I've implemented car detection so that it can avoid them, but it's not working 100%. And for some reason this model is working way better without minimap than with it :DD.

## Requirements

 * Python 3.8
 * Tensorflow
 * Keras
 * NumPy
 * OpenCV-python
 * d3dshot
 * keyboard

direct inputs\
source to this solution and code: http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game\
http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

## Installation

```bash
git clone https://github.com/CallMeRZIBI/Forza-autopilot.git
cd Forza_autopilot
mkdir collected_data
mkdir training_data
```

## Usage

For collecting data:
```bash
py data_collection.py # then enter the number of actual session when you will be promted

# To end collecting data click on the image viewer and press 'q'.
```

For loading data to dataset:
```bash
py load_data.py # this can take a while
```

For training model:
```bash
py train_model.py
```

And lastly for running trained model:
```bash
py run_model.py

# If you want to pause the model just press 'o' and for playing press 'i'.
# For total stop press 'q'.
```

## Quick Demo
https://www.youtube.com/watch?v=QuSVBz2y48c
