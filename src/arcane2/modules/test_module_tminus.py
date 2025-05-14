import datetime
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import torch
from loguru import logger
from plotly.subplots import make_subplots
from tqdm import tqdm

from ..data.data_utils.event import (
    EventCatalog,
    find,
    merge_columns_with_suffix_mean,
    overlap_with_list,
)
from ..data.data_utils.insitu_plot import plotly_plot_insitu
from .classifier_module import ClassifierModule


class TestModuleTMinus:
    def __init__(
        self,
        classifier_module: ClassifierModule,
        test_dataloader,
        device: str = "cpu",
    ):
        """
        Initializes the TestModule with the provided classifier_module and test_dataloader.

        Args:
            classifier_module: The trained ClassifierModule to use for inference.
            test_dataloader: Dataloader that loads the test set batches.
        """
        self.classifier_module = classifier_module
        self.classifier_module.eval()  # Set the classifier to evaluation mode
        self.test_dataloader = test_dataloader
        self.device = device

        self.all_results = None
        self.predicted_catalogs = {}
        self.true_catalog = None

        self.TP = {}
        self.FP = {}
        self.FN = {}
        self.detected = {}
        self.delays = {}

    def run_inference(
        self,
        df=None,
        mode="last",
        modelname=None,
        image_key="insitu",
        label_key="catalog",
        save_history_step=1,
    ):
        """
        Run inference on the entire test set and save results including historical timesteps.

        Args:
            df: Existing DataFrame to merge results into.
            mode: Placeholder for additional logic, currently unused.
            modelname: Name of the model for labeling the results.
            image_key: Key to access insitu data from the batch.
            label_key: Key to access catalog labels from the batch.
            save_history_steps: Number of preceding timesteps to save.

        Returns:
            Updates `self.all_results` with the results.
        """

        logger.info("Running inference on the test set...")

        all_results = []

        batch = next(iter(self.test_dataloader))
        logger.info(f"insitu_data.shape: {batch['insitu'].shape}")
        logger.info(f"catalog.shape: {batch['catalog'].shape}")
        logger.info(f"timestamps.shape: {batch['timestamp'].shape}")
        logger.info(f"idxs.shape: {batch['idx'].shape}")

        # Set the model to evaluation mode
        self.classifier_module.eval()

        # Disable gradient calculation for faster inference
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_dataloader)):
                insitu_data = batch[image_key]
                segmentation = batch[label_key].float().squeeze()
                timestamps = batch["timestamp"]
                idxs = batch["idx"]

                seg_hat = self.classifier_module(
                    insitu_data.to(self.classifier_module.device)
                )

                true_segmentation = segmentation.cpu()[:, -save_history_step]
                segmentation_hat = seg_hat.cpu()[:, 1:, -save_history_step]

                batch_results = [
                    (int(idx.item()), ts.item(), seg_true.item(), seg_pred.item())
                    for idx, ts, seg_true, seg_pred in zip(
                        idxs, timestamps, true_segmentation, segmentation_hat
                    )
                ]

                all_results.extend(batch_results)

        if modelname is not None:
            cols = [
                "idx",
                "timestamp",
                "true_value",
                f"predicted_value_{modelname}_tminus{save_history_step}",
            ]
            logger.info(
                f" Adding column to results: predicted_value_{modelname}_tminus{save_history_step}"
            )
        else:
            cols = [
                "idx",
                "timestamp",
                "true_value",
                "predicted_value_tminus{save_history_step}",
            ]

        all_results_df = pd.DataFrame(all_results, columns=cols)
        all_results_df["timestamp"] = pd.to_datetime(
            all_results_df["timestamp"], unit="s"
        )
        all_results_df.set_index("timestamp", inplace=True)
        all_results_df.index = all_results_df.index.round("10min")

        # Combine all_results_df and df (if df exists)
        if df is not None:
            all_results_df = pd.concat([all_results_df, df], axis=0).sort_index()
            all_results_df = all_results_df.groupby(
                all_results_df.index
            ).first()  # Keep first occurrence in case of duplicates

        # Combine self.all_results and all_results_df (if self.all_results exists)
        if self.all_results is not None:
            self.all_results = pd.concat(
                [self.all_results, all_results_df], axis=0
            ).sort_index()
            self.all_results = self.all_results.combine_first(all_results_df)
            self.all_results = self.all_results.groupby(
                self.all_results.index
            ).first()  # Keep first occurrence in case of duplicates
        else:
            self.all_results = all_results_df

    def create_catalogs(self, thresh=0.5, mean_key=None):
        if self.all_results is None:
            logger.info("No results to create catalogs from.")
            self.run_inference()

        self.predicted_catalogs = {}

        df = merge_columns_with_suffix_mean(self.all_results, mean_key=mean_key)

        keys = [col for col in df.columns if "predicted_value" in col]

        for key in keys:
            self.predicted_catalogs[key] = EventCatalog(
                event_types="Event",
                catalog_name=key.replace("predicted_value", "prediction"),
                spacecraft="Wind",
                dataframe=df,
                key=key,
                thresh=thresh,
            )

            logger.info(
                f"{key.replace("predicted_value", "prediction")} contains {len(self.predicted_catalogs[key].event_cat)} events."
            )

        self.true_catalog = EventCatalog(
            event_types="CME",
            catalog_name="True Catalog",
            spacecraft="Wind",
            dataframe=df,
            key="true_value",
        )

        logger.info(f"True Catalog contains {len(self.true_catalog.event_cat)} events.")

    def compare_catalogs(self, thresh=0.01, choice="first", duration_creepies=0):
        """
        Compare the true and predicted catalogs.
        """

        if self.true_catalog is None or self.predicted_catalogs == {}:
            logger.info("No catalogs to compare.")
            self.create_catalogs()

        keys = [col for col in self.all_results.columns if "predicted_value" in col]

        for key in keys:
            # Compare the true and predicted catalogs
            TP = []
            FP = []
            FN = []
            detected = []
            delays = []
            durations = []

            logger.info(
                f"Evaluating Early Classification: Event has to be observed for more than {duration_creepies} minutes."
            )
            creep_events = [
                x
                for x in self.predicted_catalogs[key].event_cat
                if x.duration <= datetime.timedelta(minutes=duration_creepies)
            ]

            logger.info(
                f"Removing {len(creep_events)} creep events from {key} that are shorter than/equal to{duration_creepies} minutes."
            )

            for creep in creep_events:
                self.predicted_catalogs[key].event_cat.remove(creep)

            for true_event in self.true_catalog.event_cat:
                corresponding = find(
                    true_event,
                    self.predicted_catalogs[key].event_cat,
                    thresh=thresh,
                    choice=choice,
                )

                if corresponding is None:
                    FN.append(true_event)
                else:
                    delays.append(
                        (corresponding.begin - true_event.begin).total_seconds() / 60
                    )
                    durations.append(true_event.duration)
                    TP.append(corresponding)
                    detected.append(true_event)

            FP = [
                x
                for x in self.predicted_catalogs[key].event_cat
                if max(overlap_with_list(x, self.true_catalog.event_cat, percent=True))
                == 0
            ]

            logger.info(f"TP: {len(TP)}, FP: {len(FP)}, FN: {len(FN)}")
            if len(delays) == 0:
                logger.info("No delays.")
            else:
                logger.info(f"Mean delay: {sum(delays)/len(delays)} minutes")

            self.TP[key] = TP
            self.FP[key] = FP
            self.FN[key] = FN
            self.detected[key] = detected
            self.delays[key] = delays
            self.durations[key] = durations

    def plot_dot(self, event, df, color, name, showlegend):
        """
        Plot a dot on the figure.
        """

        bmax = event.get_value(df, "bt", "max")
        vmax = event.get_value(df, "vt", "max")
        betamed = event.get_value(df, "beta", "median")

        trace1 = go.Scatter(
            x=[bmax],
            y=[betamed],
            mode="markers",
            marker=dict(color=color),
            name=name,
            showlegend=showlegend,
        )

        trace2 = go.Scatter(
            x=[bmax] if vmax != [] else [],
            y=[vmax],
            mode="markers",
            marker=dict(color=color),
            name=name,
            showlegend=False,
        )

        return trace1, trace2

    def analyse_results(self, filedir=None, df=None, catalogfig=None):
        """
        Analyse the results of the comparison between the true and predicted catalogs.
        """

        if self.TP == {} or self.FP == {} or self.FN == {} or self.detected == {}:
            logger.info("No results to analyse.")
            self.compare_catalogs()

        if df is None:
            df = self.all_results

        for catalogkey in self.predicted_catalogs.keys():
            # Analyse the results
            if catalogfig is None:
                catalogfig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                )

            catalogfig.update_yaxes(
                title_text="\u03b2" + "<sub>median</sub>", row=1, col=1
            )
            catalogfig.update_yaxes(title_text="V<sub>max</sub> [km/s]", row=2, col=1)
            catalogfig.update_xaxes(title_text="B<sub>max</sub>")

            first_tp, first_fp, first_fn = True, True, True

        for event in self.TP[catalogkey]:
            trace1, trace2 = self.plot_dot(
                event=event,
                df=df,
                color="green",
                name="True Positives",
                showlegend=first_tp,
            )
            catalogfig.add_trace(trace1, row=1, col=1)
            catalogfig.add_trace(trace2, row=2, col=1)
            first_tp = False

        for event in self.FP[catalogkey]:
            trace1, trace2 = self.plot_dot(
                event=event,
                df=df,
                color="orange",
                name="False Positives",
                showlegend=first_fp,
            )
            catalogfig.add_trace(trace1, row=1, col=1)
            catalogfig.add_trace(trace2, row=2, col=1)
            first_fp = False

        for event in self.FN[catalogkey]:
            trace1, trace2 = self.plot_dot(
                event=event,
                df=df,
                color="red",
                name="False Negatives",
                showlegend=first_fn,
            )
            catalogfig.add_trace(trace1, row=1, col=1)
            catalogfig.add_trace(trace2, row=2, col=1)
            first_fn = False

        catalogfig.update_layout(title_text=f"Results - {catalogkey}")

        if filedir is not None:
            catalogfig.write_html(str(filedir).replace(".html", f"_{catalogkey}.html"))
        else:
            catalogfig.show()

    @staticmethod
    def check_load_cache(root_path, diff_name=""):
        root_path = Path(root_path)

        if diff_name == "":
            results_path = root_path / "all_results.pkl"
        else:
            results_path = root_path / f"all_results_{diff_name}.pkl"

        if not results_path.exists():
            return False

        return True

    @staticmethod
    def load(
        root_path,
        classifier_module,
        test_dataloader,
        device="cpu",
        diff_name="",
    ):
        test_module = TestModuleTMinusOptimized(classifier_module, test_dataloader)

        test_module.device = device

        root_path = Path(root_path)

        if diff_name == "":
            results_path = root_path / "all_results.pkl"
        else:
            results_path = root_path / f"all_results_{diff_name}.pkl"

        if results_path.exists():
            with open(results_path, "rb") as path:
                test_module.all_results = pickle.load(path)

        return test_module

    def save(self, path, overwrite=True, diff_name=""):
        root_path = Path(path)
        if root_path.exists() and not overwrite:
            raise IOError(f"{root_path} already exists and not overwriting.")

        if diff_name == "":
            results_path = root_path / "all_results.pkl"
        else:
            results_path = root_path / f"all_results_{diff_name}.pkl"

        with open(results_path, "wb") as path:
            pickle.dump(self.all_results, path)

        logger.info(f"Saving all_results to: {results_path}")

    def plot_results(self, filedir=None):
        if self.all_results is None:
            logger.info("No results to plot.")
            self.run_inference()

        for year in range(
            self.all_results.index.year.min(),
            self.all_results.index.year.max() + 1,
        ):
            logger.info(f"Creating figure for year: {year}")
            year_data = self.all_results[self.all_results.index.year == year]
            subdf = year_data.filter(like="value")

            fig = plotly_plot_insitu(
                data=year_data,
                begin=year_data.index.min(),
                end=year_data.index.max(),
                variables=["B", "beta", "Targets"],
                catalogs=[self.true_catalog] + list(self.predicted_catalogs.values()),
                subdf=subdf,
            )

            if filedir is not None:
                fig.write_html(str(filedir).replace(".html", f"_{year}.html"))
            else:
                fig.show()

        return fig

    def plot_results_event(
        self,
        filedir=None,
        n_events=10,
        which="TP",
        delta=6,
        variables=["B", "beta", "Targets"],
    ):
        if self.TP == {} or self.FP == {} or self.FN == {} or self.detected == {}:
            logger.info("No results to analyse.")
            self.compare_catalogs()

        if which == "FP":
            raise NotImplementedError(
                "Plotting False Positives event-based is not implemented yet."
            )
        elif which == "FN":
            raise NotImplementedError(
                "Plotting False Negatives event-based is not implemented yet."
            )
        elif which == "TP":
            sel_catalog = self.detected
        else:
            raise ValueError(
                f"which must be one of ['FP', 'FN', 'detected'] but got {which}"
            )

        for catalogkey in sel_catalog.keys():
            for i, event in enumerate(sel_catalog[catalogkey]):
                if i == n_events:
                    break

                fig = plotly_plot_insitu(
                    data=self.all_results,
                    begin=event.begin - datetime.timedelta(hours=delta),
                    end=event.end + datetime.timedelta(hours=delta),
                    variables=variables,
                    catalogs=[self.true_catalog, self.predicted_catalogs[catalogkey]],
                    subdf=self.all_results.filter(like="value"),
                    title_text=f"Event {i+1} - {catalogkey} - Delay: {self.delays[catalogkey][i]} minutes",
                )

                if filedir is not None:
                    fig.write_html(
                        str(filedir).replace(f"{which}", f"{which}_{catalogkey}_{i}")
                    )
                else:
                    fig.show()


class TestModuleTMinusOptimized(TestModuleTMinus):
    # run super init
    def __init__(
        self,
        classifier_module: ClassifierModule,
        test_dataloader,
        device: str = "cpu",
    ):
        super().__init__(classifier_module, test_dataloader, device)

    def run_inference_all_timesteps(
        self,
        max_timestep,
        df=None,
        modelname=None,
        image_key="insitu",
        label_key="catalog",
    ):
        """
        Run inference once and collect results for all historical timesteps.

        Args:
            max_timestep: Maximum number of historical timesteps to save.
            df: Existing DataFrame to merge results into.
            modelname: Name of the model for labeling the results.
            image_key: Key to access insitu data from the batch.
            label_key: Key to access catalog labels from the batch.
        Returns:
            Updates `self.all_results` with the results.
        """

        logger.info(
            f"Running inference on the test set for all timesteps up to {max_timestep}..."
        )
        all_results = {t: [] for t in range(1, max_timestep + 1)}

        # Set the model to evaluation mode
        self.classifier_module.eval()

        # Disable gradient calculation for faster inference
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                insitu_data = batch[image_key]
                segmentation = batch[label_key].float().squeeze()
                timestamps = batch["timestamp"]
                idxs = batch["idx"]

                # Run the model inference

                seg_hat = self.classifier_module(
                    insitu_data.to(self.classifier_module.device)
                )

                true_segmentation = segmentation.cpu()[:, -1]

                for t in range(1, max_timestep + 1):
                    segmentation_hat = seg_hat.cpu()[:, 1:, -t]

                    batch_results = [
                        (int(idx.item()), ts.item(), seg_true.item(), seg_pred.item())
                        for idx, ts, seg_true, seg_pred in zip(
                            idxs, timestamps, true_segmentation, segmentation_hat
                        )
                    ]
                    all_results[t].extend(batch_results)

        combined_results = []
        for t, results in all_results.items():
            cols = [
                "idx",
                "timestamp",
                "true_value",
                (
                    f"predicted_value_{modelname}_tminus{t}"
                    if modelname
                    else f"predicted_value_tminus{t}"
                ),
            ]
            results_df = pd.DataFrame(results, columns=cols)
            results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], unit="s")
            results_df.set_index("timestamp", inplace=True)
            results_df.index = results_df.index.round("10min")
            combined_results.append(results_df)

        all_results_df = pd.concat(combined_results)

        # Merge with existing DataFrame if provided

        if df is not None:
            all_results_df = pd.concat([all_results_df, df], axis=0).sort_index()
            all_results_df = all_results_df.groupby(all_results_df.index).first()
        if self.all_results is not None:
            self.all_results = pd.concat(
                [self.all_results, all_results_df], axis=0
            ).sort_index()
            self.all_results = self.all_results.groupby(self.all_results.index).first()
        else:
            self.all_results = all_results_df

        logger.info("Inference completed for all historical timesteps.")
