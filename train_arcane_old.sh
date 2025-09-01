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


#Loop through all the years between 1998 and 2024
for year in {1998..2023}; do
    # Loop through all folds (0, 1, 2)
    for fold in {0..2}; do
      echo "###################################"
      echo "###################################"
      echo "Training for year $year, fold $fold"
      echo "###################################"
      echo "###################################"

      # Execute the Python script for each combination of year and fold
      HYDRA_FULL_ERROR=1 python3 -m scripts.train \
        --config-name config \
        +base_dataset=curated_realtime_dataset_lowres \
        +boundaries=boundaries_rtsw_${year}_${fold} \
        +collate_fns=standard_collates \
        +samplers=weighted \
        +callbacks=visualisation_classification_segmentation_inference \
        +scheduler=reduce_lr_on_plateau \
        +run_name=train_arcane_rtsw_new_bounds_new_drops \
        +model=resunetplusplus_segmentation_lowres \
        +module=nguyen_segmenter \
        +optimizer=adam

    done
done
