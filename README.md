# Project Documentation

This document provides a brief overview of how to set up and run the project code.

## Original Repository

The original development of this project was done on [this GitLab repository](https://gitlab.com/sc21sc/saulcooperman-fyp). The code was eventually transferred from there to its current location.

## Setup Instructions

### Conda Environment

To set up the required environment for running the code, use the `environment_fyp.yml` file provided in the repository to create a Conda environment:

```bash
conda env create -f environment_fyp.yml
```

This will install all necessary dependencies as specified in the YAML file.

## Running Code
To run `FILE.py`

```bash
python FILE.py
```

To run `neuron_visualisation.py` with parameters `parameters.ini`

```bash
python neuron_visualisation.py parameters.ini
```
Examples of the framework being used can be found in the .py files not in tests/ and simple_worm/


## Running Tests

To run tests, use the following command, replacing TEST_NAME with the name of the test file you wish to run:

```bash
python -m pytest tests/TEST_NAME.py
```


# Additional Documentation

For more detailed information, refer to the README_REAL.md file.