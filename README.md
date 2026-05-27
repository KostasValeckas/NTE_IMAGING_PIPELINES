# NTE IMAGING PIPELINES

This repository holds the code developed as a part of the Masters' thesis project
by Kostas Valeckas, NBI and Nordic Optical Telescope. 

## Disclaimer and License

The code is currently work in progress and any usage in research is at own 
responsibility for correctness.



## Quick description

This project contains a fully automatic data processing pipeline for astronomical
imaging in the visual and near infrared domains.

This project is developed with the goal to prepare imaging pipelines for the 
Nordic Optical Telescope Transient Explorer (NTE). As this instrument is under developtment, 
this code uses two other instruments at the Nordic Optical Telescope as proxies: 
ALFOSC (as a proxy for NTE visual imager) and NOTcam (as a proxy for NTE near-infrared 
imager).

# Installation 

## The Python code

The pipeline as it is now comes distributed as scripts. Therefore, there is no 
installation needed, as long as the code is dowloaded from this repository.

## Requirments 

run the software. If you do not know how to do so, see the [primer on Python envirorment](#how-to-create-a-clean-python-envirorment) below. 




### SExtractor, SCAMP and SWarp 

The astrometric and photometric parts of the pipeline uses [Source Extractor](https://www.astromatic.net/software/sextractor/), [SCAMP](https://www.astromatic.net/software/scamp/) and [SWarp](https://www.astromatic.net/software/swarp/). You need these tools installed if you wish to run the 
astrometric and photometric part of the pipeline. The pipeline is tested with the current versions 
of these tools: 

| Tool | Version |
|------|---------|
| Source Extractor | 2.29.0 |
| SCAMP | 2.10.0 |
| SWarp | 2.41.5 |

Follow the hyperlinks above for installation. 

Furthermore, these tools need to have a specific alias to interface correctly with 
the pipeline code. For example, some distributions of [SWarp](https://www.astromatic.net/software/swarp/) install the tool to be aliased as either `swarp` or `SWarp` depending on the version. 

For the software to work, you need to alias them as in the following:

| Tool | Command Line Alias |
|------|---------|
| Source Extractor | `sex` |
| SCAMP | `scamp` |
| SWarp | `SWarp` |

This means when you call `sex --help`, `scamp --help` and `SWarp --help` you should 
see the help pages for the tool getting printed in the command line.

### Using a docker image

The above described needed installations of [Source Extractor](https://www.astromatic.net/software/sextractor/), [SCAMP](https://www.astromatic.net/software/scamp/) and [SWarp](https://www.astromatic.net/software/swarp/) can require some configuration. Therefore, a [Docker container](https://www.docker.com/) is provided with these tools already being on-boarded so the software can be 
tested and run in a pre-configured envirorment.


```bash
sudo docker build -t nte_imaging_pipelines .
```

```bash
sudo docker run -it nte_imaging_pipelines bash
```

```bash
python3 src/run.py /app/test_data/alfosc/GRB250404A alfosc
```

```bash
python3 src/run.py /app/test_data/notcam/sn2021xel notcam
```
# How to create a clean Python envirorment

To ensure the best possible stability of the software and to avoid version conflicts with other Python packages on your system, it is **strongly recommended** to create a clean Python environment for running the software.

**Using Anaconda (conda) (recommended):**

To create a new virtual environment using Anaconda, run the following command in your terminal 
(if you are using Windows, do this and all following commands from the Anaconda Prompt):

    conda create --name NTE_IMAGING_PIPELINE python=3.11

You can replace ``NTE_IMAGING_PIPELINE`` with any name you like. This will create a new environment with Python 3.11 installed.

To activate the environment, run:

    conda activate NTE_IMAGING_PIPELINE

**Using venv (standard Python):**

To create a new virtual environment using venv (standard Python), make sure you have Python 3.11 installed,
then run the following command in your terminal:

    python3.11 -m venv NTE_IMAGING_PIPELINE

You can replace ``NTE_IMAGING_PIPELINE`` with any name you like. This will create a new environment with Python version 3.11 installed.



    If you are using Windows, you might need to run the following command instead:

        python -m venv NTE_IMAGING_PIPELINE

    This is because the Python executable might not be named ``python3.11`` on Windows.
    In that case, you can ensure that the correct version of Python is used by running:


        python --version

    If the Python version printed is not 3.11, you have several options:

    1. If your version is not 3.11, you most likely will be fine. Otherwise, try one of the following steps.
    2. Install Anaconda and create the environment using the conda command as described above.
    3. You can set the Python version to be used by the terminal by adding the Python installation directory to the PATH environment variable. See the following link for more information: `How to set the path and environment variables in Windows <https://realpython.com/add-python-to-path/>`_.

To activate the environment, run:

For Linux/MacOS:


    source NTE_IMAGING_PIPELINE/bin/activate

, where ``NTE_IMAGING_PIPELINE/bin/activate`` is the path to the activate script in the environment.

For Windows:


    # In PowerShell
    .\NTE_IMAGING_PIPELINE\Scripts\Activate.ps1


    # In cmd.exe
    .\NTE_IMAGING_PIPELINE\Scripts\Activate.bat

, where ``NTE_IMAGING_PIPELINE/Scripts`` is the path to the activate script in the environment.
