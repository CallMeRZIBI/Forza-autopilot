# Forza autopilot

AI autopilot for Forza Horizon 4

## What it does?

This autopilot drives as you learn it to drive, it's just a convolutional neural network so it probably won't avoid other vehicles... But who knows, maybye I'll change it ;)

## Requirements

 * Python 3
 * Tensorflow
 * Keras
 * NumPy
 * OpenCV-python
 * mss
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
py data_collection.py # then enter the number of actual session
```

For loading data to database:
```bash
py load_data.py
```

For training model:
```bash
py train_model.py
```

And lastly for running trained model:
```bash
py run_model.py
```