# Software for data adquisition

Given a measurement captured in the Axis Neuron Pro, this software generates a csv of that data.

## Installation

First intall the Axis Neuron Pro  (https://neuronmocap.com/content/axis-neuron-pro) and the install Python dependencies using
```
pip install --user -r requirements.txt
```

## Generate the csv

To generate the csv from a .raw observation follow the nex steps:

* Open Axis Neuron Pro and select the desired observation (in the Example folder there is a sample file).
* Inside the Python folder, execute 
```
python connect.py
```
* In the Axis Neuron Pro play the observation. The console will show the number 64 and when finished it will emit "Comm finished"

A csv, named out.csv, will be generated inside the Python folder.
