![arcanelogo](arcanelogo.png)

# ARCANE 🛰️☀️

ARCANE stands for Automatic Realtime deteCtion ANd forEcast of Interplanetary Coronal Mass Ejections. The corresponding research article has been submitted to Space Weather and a preliminary version is available on arXiv.

## Work in Progress 🔄

This work is still being developed.

## Credits ©️

This ML pipeline or parts thereof is open source and can be used by anyone for their research. However, this does not mean that credits shouldn't be given to the main developers.

If you want to use this ML pipeline or parts thereof for your own research, please provide credits to the developers by citing the following papers: ARCANE - Early Detection of Interplanetary Coronal Mass Ejections, [Automatic Detection of Interplanetary Coronal Mass Ejections in Solar Wind In Situ Data](https://doi.org/10.1029/2022SW003149)

## Setup 🛠️
### Clone repository and download data

To run this code, first clone this repository by doing:

```
git clone https://github.com/hruedisser/arcane
```

Download the [data](https://doi.org/10.6084/m9.figshare.28309295.v3)

Place the contents of **data.zip** in **arcane/data/**

Place the contents of **cache.zip** in **arcane/cache/**

### Install conda environment

Install the environment:

```
conda env create -f arcane/environment.yml
```

## Create Results 📝

To create the figures shown in Rüdisser et al. 2025 run the notebooks **scripts/notebooks/results.ipynb** and **scripts/notebooks/data-analysis.ipynb**.

## Usage 📖

To run inference on realtime data using a trained model run **realtime_arcane.sh**

This script produces the following files in **cache/arcane_server_new/**:

- **arcane_plot_now.html** and **arcane_plot_now.png**: Images showing the realtime data together with ARCANEs detection.
- **arcane_catalog_now.csv**: A CSV file containing the detected events in the realtime data.

It additionally produces the follwing files in **cache/arcane_server_multiclass/**:

- **arcane_plot_multiclass_now.html** and **arcane_plot_multiclass_now.png**: Images showing the realtime data together with ARCANEs multiclass detection.
- **arcane_catalog_now_mo.csv** and **arcane_catalog_now_sheath**: A CSV file containing the detected multiclass events in the realtime data.


To retrain ARCANE for the entire dataset, including cross-validation, run **train_arcane.sh**

To test the trained models run **test_arcane.sh**
