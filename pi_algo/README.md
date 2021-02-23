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
pip install -r pi_algo/requirements.txt
pip install -e pi_algo
# to deactive the virtual environment
deactivate
```

## Usage

Generate the performance indicators using, assuming the folder `out` has already been created:

```term
run_protocol_1 pi_algo/pi_algo/tests/data/protocol_1/input/data/test.csv out
```

The sample file is the output of the data acquisition software.

## Docker

### Build from source

_(only tested under Linux)_

Run the following command in order to create the docker image for this PI:

```console
docker build . -t pi_sbs
```
### Launch the docker image

Assuming the `pi_algo/pi_algo/tests/data/protocol_1/input/data/` contains the input data, and that the directory `out/` is **already created**, and will contain the PI output:

```shell
docker run --rm -v $PWD/pi_algo/pi_algo/tests/data/protocol_1/input/data/:/in -v $PWD/out:/out pi_sbs run_protocol_1 /in/test.csv /out
```

### Test the generate docker image

A generic testing process is proposed in [Eurobench context](https://github.com/eurobench/docker_test).
Given reference input and associate output data, it verifies that,
by calling the program with the input data, the programs generates the expected output files.
This requires `python3`.

```shell
# from the root repository
# download the generic test file
wget -O test_docker_call.py https://raw.githubusercontent.com/eurobench/docker_test/master/test_docker_call.py
# set environment variables according to this repo spec.
export TEST_PLAN=pi_algo/pi_algo/tests/test_plan.xml
export DOCKER_IMAGE=pi_sbs
# launch the script test according to the plan
python3 test_docker_call.py
```

Test plan is defined in file [test_plan.xml](pi_algo/pi_algo/tests/test_plan.xml).
