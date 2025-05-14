import datetime
import re
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


class Event:
    def __init__(
        self,
        begin: datetime.datetime,
        end: datetime.datetime,
        event_type: str = "CME",
        spacecraft: str = "Wind",
        catalog: str = "ICMECAT",
        event_id: str = None,
        proba: float = None,
        proba_max: float = None,
    ):
        if begin > end:
            logger.error(f"Begin time: {begin} is after end time: {end}")

        self.begin = begin.replace(tzinfo=None)
        self.end = end.replace(tzinfo=None)
        self.duration = self.end - self.begin

        if self.duration > datetime.timedelta(days=7):
            logger.warning(
                f"Event {event_id} has a duration of {self.duration}, which is longer than 7 days"
            )
        self.event_type = event_type
        self.spacecraft = spacecraft
        self.catalog = catalog
        self.event_id = event_id
        self.proba = proba
        self.proba_max = proba_max

    def __str__(self) -> str:
        return f"{self.event_id}: {self.begin} ---> {self.end}"

    def __call__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        """
        return True if other overlaps self during 65/100 of the time
        """
        return self.overlap(other) > 0.65 * self.duration

    def overlap(self, other):
        """return the time overlap between two events as a timedelta"""
        delta1 = min(self.end, other.end)
        delta2 = max(self.begin, other.begin)
        return max(delta1 - delta2, datetime.timedelta(0))

    def union(self, other):
        """return the union between two events as a timedelta"""
        return self.end - self.begin + other.end - other.begin - self.overlap(other)

    def intersection_over_union(self, other):
        """return the intersection over union between two events"""
        return self.overlap(other) / self.union(other)

    def get_value(self, df, feature, mode="mean"):
        col_name = [col for col in df.columns if feature in col]
        if not col_name:
            return []
        col_name = col_name[0]
        return df[col_name][self.begin : self.end].agg(mode)

    def get_percentage_data(self, df, freq="10min"):
        event_data = df[self.begin : self.end]
        event_data = event_data.dropna()
        event_range = pd.date_range(start=self.begin, end=self.end, freq=freq)
        missing_timesteps = len(event_range) - len(event_data)
        self.percentage = 1 - missing_timesteps / len(event_range)

        return self.percentage

    def get_percentile_data(self, df, freq="10min"):
        duration_quarters = self.duration / 4
        percentages = []

        for i in range(4):
            start_time = self.begin + i * duration_quarters
            end_time = start_time + duration_quarters if i < 3 else self.end

            event_data = df[start_time:end_time]
            event_range = pd.date_range(start=start_time, end=end_time, freq=freq)

            missing_timesteps = len(event_range) - len(event_data)
            percentage = 1 - missing_timesteps / len(event_range)

            percentages.append(percentage)

        return tuple(percentages)

    def get_event_start_data(self, df, freq="10min", hours=3):
        event_data = df[self.begin : self.begin + datetime.timedelta(hours=hours)]
        event_range = pd.date_range(
            start=self.begin,
            end=self.begin + datetime.timedelta(hours=hours),
            freq=freq,
        )
        missing_timesteps = len(event_range) - len(event_data)
        percentage_start_data = 1 - missing_timesteps / len(event_range)
        self.percentage_start_data = percentage_start_data
        return percentage_start_data

    def get_data(self, df, delta=6, delta2=0):
        if delta2 == 0:
            delta2 = delta
        else:
            delta2 = delta2

        return df[
            self.begin
            - datetime.timedelta(hours=delta) : self.end
            + datetime.timedelta(hours=delta2)
        ]

    def plot_mag(self, df, delta=6, pred_key="", true_key=""):
        data = self.get_data(df, delta)

        bt_keys = [col for col in data.columns if "bt" in col]
        bx_keys = [col for col in data.columns if "bx" in col]
        by_keys = [col for col in data.columns if "by" in col]
        bz_keys = [col for col in data.columns if "bz" in col]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(data.index, data[bt_keys[0]], label="|B|")
        axes[0].plot(data.index, data[bx_keys[0]], label="Bx")
        axes[0].plot(data.index, data[by_keys[0]], label="By")
        axes[0].plot(data.index, data[bz_keys[0]], label="Bz")

        axes[1].plot(data.index, data[pred_key], label="Pred")

        if true_key != "":
            axes[1].plot(data.index, data[true_key], label="True")

        axes[0].axvline(self.begin, color="r", linestyle="--", label="Event Start")
        axes[0].axvline(self.end, color="r", linestyle="--", label="Event End")

        # format x-axis as YYYY-MM-DD
        axes[0].xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
        )
        axes[0].xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=1))

        plt.tight_layout()
        plt.show()

        return fig, axes


class DONKIEvent(Event):
    def __init__(
        self,
        event_type: str = "CME",
        spacecraft: str = "Wind",
        catalog: str = "ICMECAT",
        event_id: str = None,
        proba: float = None,
        proba_max: float = None,
        launch_time: datetime.datetime = None,
        arrival_time_err: datetime.timedelta = None,
        arrival_time: datetime.datetime = None,
        initial_speed: float = None,
        longitude: float = None,
        latitude: float = None,
    ):
        begin = arrival_time - arrival_time_err if arrival_time else None
        end = arrival_time + arrival_time_err if arrival_time else None

        super().__init__(
            begin=begin,
            end=end,
            event_type=event_type,
            spacecraft=spacecraft,
            catalog=catalog,
            event_id=event_id,
            proba=proba,
            proba_max=proba_max,
        )
        self.launch_time = launch_time
        self.initial_speed = initial_speed
        self.arrival_time = arrival_time
        self.arrival_time_err = arrival_time_err
        self.longitude = longitude
        self.latitude = latitude


class EventCatalog:
    def __init__(
        self,
        folder_paths: List[Union[str, Path, None]] = [],
        event_types: str = "CME",
        catalog_name: str = "ICMECAT",
        spacecraft: str = "Wind",
        startname: str = "icme_start_time",
        endname: str = "mo_end_time",
        dataframe: pd.DataFrame = None,
        key: str = None,
        resample_freq: str = "30min",
        creep_delta: int = 20,
        thresh: float = 0.5,
    ):
        self.event_types = event_types
        self.catalog_name = catalog_name
        self.spacecraft = spacecraft
        self.startname = startname
        self.endname = endname
        self.dataframe = dataframe
        self.key = key
        self.resample_freq = resample_freq
        self.creep_delta = creep_delta
        self.thresh = thresh

        if len(folder_paths) > 0:
            try:
                if folder_paths[0].startswith("http"):
                    self.folder_paths = folder_paths
                else:
                    self.folder_paths = [
                        Path(folder_path) for folder_path in folder_paths
                    ]
            except Exception:
                self.folder_paths = [Path(folder_path) for folder_path in folder_paths]

            logger.info(
                f"Reading {self.catalog_name} catalog from {self.folder_paths}: {self.event_types} events for {self.spacecraft}"
            )
            self.event_cat = self.read_catalog()
            logger.info(
                f"Read {len(self.event_cat)} events from {self.catalog_name} catalog"
            )

        else:
            logger.info(
                f"Creating {self.catalog_name} catalog from dataframe at key: {key}"
            )
            self.event_cat = self.create_catalog()
            logger.info(
                f"Created {len(self.event_cat)} events in {self.catalog_name} catalog"
            )

    def __str__(self) -> str:
        return f"{self.catalog_name} catalog: {self.event_types} - {self.spacecraft}"

    def __call__(self) -> str:
        return self.__str__()

    def read_catalog(self) -> List[Event]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.event_cat)

    @property
    def first_event(self):
        return self.event_cat[0] if self.event_cat else None

    @property
    def last_event(self):
        return self.event_cat[-1] if self.event_cat else None

    @property
    def begin(self):
        return self.first_event.begin if self.first_event else None

    @property
    def end(self):
        return self.last_event.end if self.last_event else None

    def filter_catalog(
        self, begin: datetime.datetime, end: datetime.datetime
    ) -> List[Event]:
        """
        Filter events in catalog between begin and end
        """
        return [
            event
            for event in self.event_cat
            if event.end >= begin and event.begin <= end
        ]

    def create_catalog(self) -> List[Event]:
        logger.info(
            f"Creating {self.catalog_name} catalog from dataframe at key: {self.key}"
        )
        # fill with 0
        labels = (
            self.dataframe[self.key].resample(self.resample_freq).asfreq().fillna(0)
        )

        positive_labels = (labels > self.thresh).astype(int)
        if sum(positive_labels) == 0:
            logger.warning(f"No positive {self.key} found in dataframe")
            return []

        changepoints = positive_labels.diff()

        if positive_labels.iloc[0] == 1:
            changepoints.iloc[0] = 1
        if positive_labels.iloc[-1] == 1:
            changepoints.iloc[-1] = -1

        begins = changepoints.index[changepoints == 1]
        ends = changepoints.index[changepoints == -1]

        if len(begins) == 0 or len(ends) == 0:
            logger.warning(f"No events found in {self.key}")
            return []

        # Ensure begins and ends are aligned
        if ends[0] < begins[0]:
            begins = begins.insert(0, labels.index[0])
        if begins[-1] > ends[-1]:
            ends = ends.append(pd.Index([labels.index[-1]]))

        try:
            evtlist_full = [
                Event(
                    begin=begins[i],
                    end=ends[i],
                    event_type=self.event_types,
                    spacecraft=self.spacecraft,
                    catalog=self.catalog_name,
                    event_id=f"{self.catalog_name}_{self.event_types}_{i+1}",
                )
                for i in range(len(ends))
            ]
        except IndexError:
            evtlist_full = [
                Event(
                    begin=begins[i],
                    end=ends[i],
                    event_type=self.event_types,
                    spacecraft=self.spacecraft,
                    catalog=self.catalog_name,
                    event_id=f"{self.catalog_name}_{self.event_types}_{i+1}",
                )
                for i in range(len(ends) - 1)
            ]

        if len(evtlist_full) == 0:
            logger.warning(f"No events found in {self.key}")
            return []

        # Merge events that are closer than `creep_delta` minutes apart
        merged_evtlist = []
        creep_delta = datetime.timedelta(minutes=self.creep_delta)

        current_event = evtlist_full[0]
        for next_event in evtlist_full[1:]:
            # If the gap between the current event's end and the next event's begin is smaller than creep_delta, merge them
            if next_event.begin - current_event.end <= creep_delta:
                # Extend the current event's end to the next event's end
                current_event.end = next_event.end
            else:
                # If not, add the current event to the list and start a new one
                merged_evtlist.append(current_event)
                current_event = next_event

        # Add the last event
        merged_evtlist.append(current_event)

        # Filter out events shorter than or equal to the creep_delta
        merged_evtlist_filtered = [
            event
            for event in merged_evtlist
            if event.duration > datetime.timedelta(minutes=self.creep_delta)
        ]

        return merged_evtlist_filtered


def is_in_list(ref_event, event_list, thresh, percent=True):
    """
    returns True if ref_event is overlapped thresh percent of its duration by
    at least one elt in event_list
    """

    if percent:
        return max(overlap_with_list(ref_event, event_list, percent=True)) > thresh

    else:
        return (
            max(overlap_with_list(ref_event, event_list)) > thresh * ref_event.duration
        )


def find(ref_event, event_list, thresh=0.01, choice="first"):
    """
    Return the event in event_list that overlap ref_event for a given threshold
    if it exists
    Choice give the preference of returned :
    first return the first of th
    Best return the one with max overlap
    merge return the combination of all of them
    """

    if is_in_list(ref_event, event_list, thresh):
        return chose_event_from_list(ref_event, event_list, thresh, choice)
    else:
        return None


def find_iou(ref_event, event_list, thresh=0.5):
    """
    Return the event in event_list that overlap ref_event for a given threshold
    """

    if any([ref_event.intersection_over_union(elt) > thresh for elt in event_list]):
        return event_list[
            np.argmax([ref_event.intersection_over_union(elt) for elt in event_list])
        ]
    else:
        return None


def find_equal(ref_event, event_list, uncertainty: int):
    """
    Check if an event exists in the event_list that matches ref_event within a given uncertainty.

    Parameters:
    ref_event (Event): The reference event to compare.
    event_list (List[Event]): A list of events to search for matches.
    uncertainty (int): Allowed uncertainty in minutes for start and end times.

    Returns:
    Event or None: Returns the matching event if found, otherwise None.
    """
    uncertainty_delta = datetime.timedelta(minutes=uncertainty)

    for event in event_list:
        start_diff = abs(ref_event.begin - event.begin)
        end_diff = abs(ref_event.end - event.end)

        if start_diff <= uncertainty_delta and end_diff <= uncertainty_delta:
            return event

    return None


def similarity(event1, event2):
    if event1 is None:
        return 0
    inter = event1.overlap(event2)
    return inter / (event1.duration + event2.duration - inter)


def overlap_with_list(ref_event, event_list, percent=True):
    """
    return the list of the overlaps between an event and the elements of
    an event list
    Have the possibility to have it as the percentage of fthe considered event
    in the list
    """
    if percent:
        return [ref_event.overlap(elt) / ref_event.duration for elt in event_list]
    else:
        return [ref_event.overlap(elt) for elt in event_list]


def chose_event_from_list(ref_event, event_list, thresh, choice="first", percent=True):
    """
    return an event from even_list according to the choice adopted
    first return the first of the lists
    last return the last of the lists
    best return the one with max overlap
    merge return the combination of all of them
    """
    if choice == "first":
        return event_list[
            np.argmax(
                [x > thresh for x in overlap_with_list(ref_event, event_list, percent)]
            )
        ]
    if choice == "last":
        return event_list[-1]
    if choice == "best":
        return event_list[np.argmax(overlap_with_list(ref_event, event_list, percent))]
    if choice == "merge":
        return merge(event_list[0], event_list[-1])


def merge(event1, event2):
    return Event(
        event1.begin,
        event2.end,
        event_type=event1.event_type,
        spacecraft=event1.spacecraft,
        catalog=event1.catalog,
        event_id=event1.event_id + "-" + event2.event_id,
    )


def merge_sheath(event1, event2):
    return Event(
        event2.begin,
        event1.end,
        event_type=event1.event_type,
        spacecraft=event1.spacecraft,
        catalog=event1.catalog,
        event_id=event1.event_id + "-" + event2.event_id,
    )


def eval_key_params(eventlist, df, keys, minutes):
    true = {}
    pred = {}
    diff = {}

    for key in keys:
        true[key] = []
        pred[key] = []
        diff[key] = []

        for event in eventlist:
            subdf = df[(df.index >= event.begin) & (df.index <= event.end)]
            small_df = subdf[
                subdf.index <= event.begin + datetime.timedelta(minutes=minutes)
            ]

            truekey = f"true_{key}"
            predkey = f"pred_{key}"

            true[key].append(small_df[truekey].values[-1])
            pred[key].append(small_df[predkey].values[-1])
            diff[key].append(
                small_df[truekey].values[-1] - small_df[predkey].values[-1]
            )

    return true, pred, diff


def merge_columns_with_suffix_mean(df, mean_key=None):
    # Create a dictionary to hold the base name and corresponding columns
    columns_dict = {}

    if mean_key:
        for key in mean_key:
            columns = [col for col in df.columns if col.startswith(key)]
            if columns:
                columns_dict[key] = columns

    else:
        # Regular expression to match columns with the same base name followed by a number
        pattern = re.compile(r"(.+)_\d+$")
        # Iterate through the columns and group them by their base names
        for col in df.columns:
            match = pattern.match(col)
            if match:
                base_name = match.group(1)
                if base_name in columns_dict:
                    columns_dict[base_name].append(col)
                else:
                    columns_dict[base_name] = [col]

    # For each base name, calculate the mean of its corresponding columns and add a new column
    for base_name, columns in columns_dict.items():
        # Compute the row-wise mean of the columns
        df[base_name] = df[columns].mean(axis=1)
        # Drop the original columns
        df.drop(columns=columns, inplace=True)

    return df


def cat_per_threshold(df, key, duration=30, threshs=0.001):
    thresholds = np.arange(df[key].min(), df[key].max(), threshs).tolist()
    cat_dict = {}

    for thresh in thresholds:
        # Create the EventCatalog for the given threshold
        cat = EventCatalog(
            event_types="CME",
            spacecraft="Wind",
            dataframe=df,
            key=key,
            creep_delta=duration,
            thresh=thresh,
        )

        # Filter out events shorter than/equal to the duration
        original_count = len(cat.event_cat)
        cat.event_cat = [
            event
            for event in cat.event_cat
            if event.duration > datetime.timedelta(minutes=duration)
        ]
        removed_count = original_count - len(cat.event_cat)

        logger.info(
            f"Removing {removed_count} creep events from {key} that are shorter than/equal to {duration} minutes."
        )
        logger.info(
            f"Catalog for threshold {thresh} contains {len(cat.event_cat)} events."
        )

        # Save the filtered catalog
        cat_dict[thresh] = cat.event_cat

    return cat_dict


def compare_catalogs(true_cat, pred_cat, thresh=0.01, choice="first"):
    TP = []
    FP = []
    FN = []
    detected = []
    delays = []
    durations = []
    found_already = []

    if pred_cat == []:
        FN = true_cat
        logger.info("No events detected.")
    else:
        for true_event in true_cat:
            corresponding = find(true_event, pred_cat, thresh=thresh, choice=choice)

            if corresponding is None:
                FN.append(true_event)
            else:
                TP.append(corresponding)
                detected.append(true_event)
                if corresponding in found_already:
                    delays.append(0)
                else:
                    delays.append((corresponding.begin - true_event.begin).seconds / 60)
                    found_already.append(corresponding)

    if true_cat == []:
        FP = pred_cat
    else:
        FP = [x for x in pred_cat if max(overlap_with_list(x, true_cat)) == 0]

    return TP, FP, FN, detected, delays, durations
