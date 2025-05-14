#!/bin/bash -l

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Name of the environment and path to the env file
ENV_NAME="lightning-arcane-env"
ENV_FILE="environment.yml"

# Check if the environment already exists
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' not found. Creating it from $ENV_FILE..."
    conda env create -f "$ENV_FILE"
fi

# Activate environment
conda activate "$ENV_NAME"

echo "###################################"
echo "###################################"
echo " Running ARCANE for Realtime Data "
echo "###################################"
echo "###################################"

python3 -m scripts.predict_server_new \
  --config-name config_server \
  +base_dataset=curated_realtime_dataset_lowres_helcats \
  +boundaries=boundaries_server \
  +collate_fns=standard_collates_helcats \
  +samplers=weighted \
  +callbacks=visualisation_classification_segmentation_inference \
  +scheduler=reduce_lr_on_plateau \
  +run_name=arcane_server_new \
  +model=resunetplusplus_segmentation_lowres \
  +module=nguyen_segmenter \
  +optimizer=adam

echo '--------------------------------------------'
echo

python3 -m scripts.predict_server_multiclass \
  --config-name config_server \
  +base_dataset=curated_realtime_dataset_lowres_multiclass \
  +boundaries=boundaries_server \
  +collate_fns=multiclass_collates \
  +samplers=weighted \
  +callbacks=visualisation_classification_segmentation_multiclass \
  +scheduler=reduce_lr_on_plateau \
  +run_name=arcane_server_multiclass \
  +model=resunetplusplus_segmentation_lowres_multiclass \
  +module=nguyen_segmenter_multiclass \
  +optimizer=adam

echo '--------------------------------------------'
echo
