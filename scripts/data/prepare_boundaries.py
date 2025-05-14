from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm


def create_boundaries(years, test_year, n):
    """
    Splits a list of years into train, validation, and test groups.

    Args:
        years (list): A list of years.
        test_year (int): The year to exclude for testing.
        n (int): Determines how to assign the groups: 0 means the third group is validation,
                 1 means the second group is validation.
                 2 means the first group is validation.

    Returns:
        train_boundaries (list of tuples): Train boundaries in the format [["start", "end"]].
        val_boundaries (list of tuples): Validation boundaries in the format [["start", "end"]].
        test_boundaries (list of tuples): Test boundaries containing the excluded test year.
    """

    # Step 1: Filter out the test_year
    remaining_years = [year for year in years if year != test_year]

    # Step 2: Split the remaining years into three equal equal groups

    if len(remaining_years) % 3 == 2:
        group_size = (len(remaining_years) // 3) + 1
    else:
        group_size = len(remaining_years) // 3

    group1 = remaining_years[:group_size]
    group2 = remaining_years[group_size : group_size * 2]
    group3 = remaining_years[group_size * 2 :]

    if n == 0:
        logger.info(f"Group 1: {group1}")
        logger.info(f"Group 2: {group2}")
        logger.info(f"Group 3: {group3}")

    # Step 3: Assign the groups based on the value of `n`
    if n == 0:
        train_years = group1 + group2
        val_years = group3
    elif n == 1:
        train_years = group1 + group3
        val_years = group2
    elif n == 2:
        train_years = group2 + group3
        val_years = group1
    else:
        raise ValueError("n should be 0, 1 or 2.")

    # Step 4: Create the test boundaries
    test_boundaries = [[f"{test_year}0101T000000", f"{test_year+1}0101T000000"]]

    # Step 5: Helper function to create boundaries for a group of years
    def create_group_boundaries(years_group):
        if not years_group:
            return []
        boundaries = []
        start_year = years_group[0]
        end_year = years_group[0]
        for year in years_group[1:]:
            if year - end_year == 1:
                end_year = year
            else:
                boundaries.append(
                    [f"{start_year}0101T000000", f"{end_year+1}0101T000000"]
                )
                start_year = year
                end_year = year
        boundaries.append([f"{start_year}0101T000000", f"{end_year+1}0101T000000"])
        return boundaries

    # Step 6: Create the boundaries for train and validation
    train_boundaries = create_group_boundaries(train_years)
    val_boundaries = create_group_boundaries(val_years)

    return train_boundaries, val_boundaries, test_boundaries


def save_yaml(file_path, data):
    """
    Save the data to a yaml file at the given path.

    Args:
        file_path (Path): The path to the yaml file.
        data (dict): The data to be written to the file.
    """
    with file_path.open("w") as yaml_file:
        yaml.dump(data, yaml_file)


def generate_yaml_files(years, output_dir, file_base_name):

    for test_year in tqdm(range(years[0], years[-1] + 1)):
        for n in range(3):  # Create for n=0,1,2
            train_boundaries, val_boundaries, test_boundaries = create_boundaries(
                years, test_year, n
            )

            # Construct the data in the required format
            yaml_data = {
                "train_boundaries": train_boundaries,
                "val_boundaries": val_boundaries,
                "test_boundaries": test_boundaries,
            }

            # File name format: {test_year}_{n}.yaml
            file_name = f"boundaries_{file_base_name}_{test_year}_{n}.yaml"
            file_path = output_dir / file_name
            logger.info(f"Saving {file_name}")

            # Save the yaml file
            save_yaml(file_path, yaml_data)


if __name__ == "__main__":

    years_omni = [1995 + i for i in range(29)]
    logger.info(
        f"Generating boundaries from {years_omni[0]} to {years_omni[-1]} for OMNI data"
    )
    output_dir = Path("config/boundaries")
    generate_yaml_files(years_omni, output_dir, file_base_name="omni")

    years_rtsw = [1998 + i for i in range(27)]
    logger.info(
        f"Generating boundaries from {years_rtsw[0]} to {years_rtsw[-1]} for RTSW data"
    )
    output_dir = Path("config/boundaries")
    generate_yaml_files(years_rtsw, output_dir, file_base_name="rtsw")
