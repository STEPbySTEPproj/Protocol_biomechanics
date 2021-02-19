# Performance indicators algorithm

## Purposes

Code to produce the Stet by Step project performance indicators.
In the `/tests/data/protocol_1_PI_Algo/input/Models` folder there are three models: one to classify if the sequence is ascend or descend and another two to classify the events based on the sequence type.
In the `/tests/data/protocol_1_PI_Algo/input/Data` folder there are `csv` files that can be used to test the code.

## Installation

### using conda

Create conda enviroment running

```term
conda create -n Env_PI_Algo python=3.6
```

and install dependencies using

```term
pip install --user -r requirements.txt
```

### using venv

```term
# From this folder
python3 -m venv venv
# to be executed everytime a terminal session is launched
source venv/bin/activate
pip install --upgrade pip
pip install -r PI_Algo/requirements.txt
# to deactive the virtual environment
deactivate
```

## Usage

Generate the performance indicators using, assuming the folder `out` has already been created:

```term
python run_protocol_1.py tests/data/protocol_1/input/data/test.csv out
```

The sample file is the output of the data acquisition software.
