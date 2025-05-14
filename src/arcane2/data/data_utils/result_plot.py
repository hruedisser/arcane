import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def plot_confusion_matrix(
    tp, fp, fn, mean_delay, test_duration, threshold, model, base_path
):
    # Create confusion matrix without TN
    cm = np.array([[tp, fp], [fn, 0]])

    # Increase figure size to accommodate more text
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.5)

    # Annotate the matrix with TP, FP, FN, and the values
    labels = [["TP", "FP"], ["FN", ""]]
    for i in range(2):
        for j in range(2):
            if cm[i, j] > 0 or (
                i == 1 and j == 1
            ):  # Include an empty label in the bottom-right
                ax.text(
                    x=j,
                    y=i,
                    s=f"{labels[i][j]}: {cm[i, j]}" if cm[i, j] > 0 else "",
                    va="center",
                    ha="center",
                    fontsize=14,
                )

    # Adjust title to avoid cutting off and wrap long titles using '\n'
    plt.title(
        f"Model: {model}\nDuration: {test_duration} min, Threshold: {threshold}\nMean Delay: {mean_delay:.2f} minutes",
        fontsize=14,
        pad=20,
    )

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout to fit everything
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        f"{base_path}/confusion_matrix_{model}_{test_duration}min_{threshold}.png",
        bbox_inches="tight",
    )


def plot_delay_distribution(dfs, base_path):
    # Iterate through each test_duration
    for duration, dur_df in dfs.items():

        for model, model_df in dur_df.items():
            TP_full = model_df["predicted"] - model_df["FP"]

            # Calculate precision, recall, and F1-score for each threshold
            model_df["Precision"] = (
                TP_full / model_df["predicted"]
            )  # model_df['TP'] / (model_df['TP'] + model_df['FP'])
            model_df["Recall"] = model_df["TP"] / (model_df["TP"] + model_df["FN"])
            model_df["F1"] = (
                2
                * (model_df["Precision"] * model_df["Recall"])
                / (model_df["Precision"] + model_df["Recall"])
            )

            # Select the best threshold (max F1-score)
            best_threshold_idx = model_df["F1"].idxmax()
            best_row = model_df.loc[best_threshold_idx]

            # Extract delays and durations for the best threshold
            delays = np.array(best_row["delays"])
            durations = np.array(best_row["durations"])
            durations_in_minutes = np.array([d.total_seconds() / 60 for d in durations])

            # First plot: Delay distribution in absolute time
            plt.figure(figsize=(10, 6))
            sns.histplot(delays, bins=200, kde=True)
            plt.title(
                f"{model} - Delay Distribution (Best Threshold: {best_threshold_idx})"
            )
            plt.xlabel("Delay (minutes)")
            plt.ylabel("Frequency")
            plt.savefig(f"{base_path}/{model}_{duration}_delay_distribution.png")
            # plt.show()

            # Second plot: Delay as a percentage of the duration
            delay_percentage = (delays / durations_in_minutes) * 100

            plt.figure(figsize=(10, 6))
            sns.histplot(delay_percentage, bins=200, kde=True)
            plt.title(
                f"{model} - Delay as Percentage of Duration (Best Threshold: {best_threshold_idx})"
            )
            plt.xlabel("Delay as Percentage of Duration (%)")
            plt.ylabel("Frequency")
            plt.savefig(
                f"{base_path}/{model}_{duration}_delay_percentage_distribution.png"
            )
            # plt.show()


def plot_precision_recall_curves(dfs, base_path, duration=30):
    # Iterate through each test_duration
    durations = list(dfs.keys())

    for duration in durations:

        fig = go.Figure()

        dur_df = dfs[duration]

        for model in dur_df.keys():
            model_df = dur_df[model]

            TP_full = model_df["predicted"] - model_df["FP"]

            model_df["recall"] = model_df["TP"] / (model_df["TP"] + model_df["FN"])
            model_df["precision"] = TP_full / (model_df["predicted"])

            fig.add_trace(
                go.Scatter(
                    x=model_df["recall"],
                    y=model_df["precision"],
                    mode="lines+markers",
                    name=f"{model}",
                    hovertemplate=(
                        f"Model: {model}<br>"
                        "Precision: %{y:.2f}<br>"
                        "Recall: %{x:.2f}<br>"
                        "%{text}"
                    ),
                    text=[
                        f'"Thresholds: {thresh:.2f} <br> Delay: mean - {model_df['mean_delay'][thresh]:.2f} min <br> std - {model_df['std_delay'][thresh]:.2f}'
                        for thresh in model_df.index
                    ],
                )
            )

        fig.update_layout(
            title=f"Precision-Recall Curve for Test Duration: {duration}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            hovermode="closest",
        )

        # Save the plot
        fig.write_html(f"{base_path}/precision_recall_{duration}min.html")
