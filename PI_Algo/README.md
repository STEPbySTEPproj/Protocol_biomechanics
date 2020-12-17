# Performance indicators algorithm

## Purposes

Code to produce the Stet by Step project performance indicators. In the /tests/data/protocol_1_PI_Algo/input/Models folder there are three models: one to classify if the sequence is ascend or descend and another two to classify the events based on the sequence type. In the /tests/data/protocol_1_PI_Algo/input/Data folder there is csv that can be used to test the code.

## Installation

Create conda enviroment running
```
conda create -n Env_PI_Algo python=3.6
```

and install dependencies using
```
pip --user install -r requirements.txt
```

## Usage
Generate the performance indicators using
```
python run_protocol_1.py sample_file_fullpath output_directorypath
```
The sample file is the output of the data acquisition software.

