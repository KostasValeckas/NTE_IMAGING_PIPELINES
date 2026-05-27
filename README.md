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

## Installation 

### The Python code


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
