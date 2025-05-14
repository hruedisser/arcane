import datetime
import sys
import time

import numpy as np
import pandas as pd
import py3dcore
from py3dcore.methods.abc_smc import abc_smc_worker
from py3dcore.methods.custom_observer import CustomObserver
from py3dcore.methods.data import FittingData
from py3dcore.methods.heliosat_utils import sanitize_dt
from py3dcore.methods.method import BaseMethod
from py3dcore.models.toroidal import ToroidalModel

from src.arcane2.core_utils.plotting_utils import plot_insitu
from src.arcane2.data.data_utils.core_data import CoreData
from src.arcane2.data.data_utils.event import DONKIEvent, Event


def starmap(func, args):
    return [func(*_) for _ in args]


import multiprocess as mp  # ing as mp

manager = mp.Manager()
processes = []


def get_modelkwargs_ranges(fittingstate_values):

    ensemble_size = fittingstate_values[0]

    model_kwargs = {
        "ensemble_size": ensemble_size,  # 2**17
        "iparams": {
            "cme_longitude": {
                "maximum": fittingstate_values[1][1],
                "minimum": fittingstate_values[1][0],
            },
            "cme_latitude": {
                "maximum": fittingstate_values[2][1],
                "minimum": fittingstate_values[2][0],
            },
            "cme_inclination": {
                "distribution": "uniform",
                "maximum": fittingstate_values[3][1],
                "minimum": fittingstate_values[3][0],
            },
            "cme_diameter_1au": {
                "maximum": fittingstate_values[4][1],
                "minimum": fittingstate_values[4][0],
            },
            "cme_aspect_ratio": {
                "maximum": fittingstate_values[5][1],
                "minimum": fittingstate_values[5][0],
            },
            "cme_launch_radius": {
                "distribution": "uniform",
                "maximum": fittingstate_values[6][1],
                "minimum": fittingstate_values[6][0],
            },
            "cme_launch_velocity": {
                "maximum": fittingstate_values[7][1],
                "minimum": fittingstate_values[7][0],
            },
            "t_factor": {
                "maximum": fittingstate_values[11][1],
                "minimum": fittingstate_values[11][0],
            },
            "cme_expansion_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[8][1],
                "minimum": fittingstate_values[8][0],
            },
            "magnetic_decay_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[12][1],
                "minimum": fittingstate_values[12][0],
            },
            "magnetic_field_strength_1au": {
                "maximum": fittingstate_values[13][1],
                "minimum": fittingstate_values[13][0],
            },
            "background_drag": {
                "distribution": "uniform",
                "maximum": fittingstate_values[9][1],
                "minimum": fittingstate_values[9][0],
            },
            "background_velocity": {
                "distribution": "uniform",
                "maximum": fittingstate_values[10][1],
                "minimum": fittingstate_values[10][0],
            },
        },
    }

    for param, values in model_kwargs["iparams"].items():
        if values["maximum"] == values["minimum"]:
            values["distribution"] = "fixed"
            values["default_value"] = values["minimum"]
            del values["maximum"]
            del values["minimum"]

    return model_kwargs


def run_automatic_fit(
    event: Event, df: pd.DataFrame, elevo_event: DONKIEvent, t: int
) -> None:

    template = "none"
    bg_color = "rgba(0, 0,0, 0)"
    line_color = "black"
    line_colors = ["#c20078", "#f97306", "#069af3", "#000000"]
    eventshade = "LightSalmon"

    lw_insitu = 2  # linewidth for plotting the in situ data
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    lw_fitp = 2  # linewidth for plotting the lines where fitting points

    event_data = event.get_data(df, delta=6)

    core_data = CoreData(
        observer="RTSW",
        reference_frame="GSM",
        df=event_data,
    )

    t_data, b_data, pos_data = core_data.get_data("HEEQ")

    observer = CustomObserver(observer=core_data.spacecraft)

    modelstatevar_ranges = [int(2**16)] + [  # ensemblesize
        [elevo_event.longitude - 20, elevo_event.longitude + 20],  # Longitude (HEEQ)
        [elevo_event.latitude - 10, elevo_event.latitude + 10],  # Latitude (HEEQ)
        [0.0, 360.0],  # Inclination
        [0.06, 0.3],  # Diameter 1 AU
        [1.0, 3.5],  # Aspect Ratio
        [21.5, 21.5],  # Launch Radius
        [
            elevo_event.initial_speed - 100,
            elevo_event.initial_speed + 100,
        ],  # Launch Velocity
        [1.0, 2.0],  # Expansion Rate
        [0.20, 3.00],  # Background Drag
        [100.0, 900.0],  # Background Velocity
        [-250.0, 250.0],  # T_Factor
        [1.0, 2.0],  # Magnetic Decay Rate
        [5.0, 65.0],  # Magnetic Field Strength 1 AU
    ]

    model_kwargs = get_modelkwargs_ranges(modelstatevar_ranges)
    iter_i = 0  # keeps track of iterations
    hist_eps = []  # keeps track of epsilon values
    hist_time = []  # keeps track of time

    balanced_iterations = 3
    time_offsets = [0]
    eps_quantile = 0.25
    epsgoal = 0.25
    kernel_mode = "cm"
    random_seed = 42
    summary_type = "norm_rmse"
    fit_coord_system = "HEEQ"
    sc = core_data.spacecraft

    multiprocessing = False
    njobs = 4

    itermin = 12
    itermax = 15
    time_limit = 20

    n_particles = 512

    print(
        f"The fitting will be run for a maximum of {itermax} iterations and a maximum of {time_limit} minutes. If the RMSE is below {epsgoal} the fitting will terminate after {itermin} iterations."
    )

    ############################
    #### Set fitting points ####
    ############################

    event_duration = (event.end - event.begin).total_seconds() / 3600

    assumed_duration = 25  # hours

    if t == 1:
        t_fits = [
            event.begin + datetime.timedelta(minutes=30),
            event.end,
        ]  # fitting point is set to the end of the event
        print(
            f"Two fitting points set to 30 minutes after start and the last available data point: {t_fits}"
        )
        print(f"Assuming a duration of {assumed_duration} hours")
        t_s = event.begin
        t_e = event.begin + datetime.timedelta(hours=assumed_duration)

    elif t == 2:
        t_fits = [
            event.begin + datetime.timedelta(hours=1),
            event.end,
        ]
        print(
            f"Two fitting points set to 1 hour after start and the last available data point: {t_fits}"
        )
        print(f"Assuming a duration of {assumed_duration} hours")
        t_s = event.begin
        t_e = event.begin + datetime.timedelta(hours=assumed_duration)

    elif t == 3:
        t_fits = [
            event.begin + datetime.timedelta(hours=1),
            event.begin + datetime.timedelta(hours=2),
            event.end,
        ]
        print(
            f"Three fitting points set to 1 hour, 2 hours after start and the last available data point: {t_fits}"
        )
        print(f"Assuming a duration of {assumed_duration} hours")

        t_s = event.begin
        t_e = event.begin + datetime.timedelta(hours=assumed_duration)

    elif t == 6:
        t_fits = [
            event.begin + datetime.timedelta(hours=1),
            event.begin + datetime.timedelta(hours=3),
            event.end,
        ]
        print(
            f"Three fitting points are set to 1 hour, 3 hours after start and the last available data point: {t_fits}"
        )
        print(f"Assuming a duration of {assumed_duration} hours")

        t_s = event.begin
        t_e = event.begin + datetime.timedelta(hours=assumed_duration)

    elif t == 12:
        t_fits = [
            event.begin + datetime.timedelta(hours=3),
            event.begin + datetime.timedelta(hours=6),
            event.begin + datetime.timedelta(hours=9),
            event.end,
        ]
        print(
            f"Four fitting points are set to 3, 6, 9 hours after start and the last available data point: {t_fits}"
        )
        print(f"Assuming a duration of {assumed_duration} hours")

        t_s = event.begin
        t_e = event.begin + datetime.timedelta(hours=assumed_duration)

    elif t == 25:
        nr_t_fits = 5

        t_s = event.begin
        t_e = event.begin + datetime.timedelta(hours=assumed_duration)

        event_duration = (event.end - event.begin).total_seconds() / 3600

        difference = event_duration / (nr_t_fits + 1)

        t_fits = [
            event.begin + datetime.timedelta(hours=difference * (i + 1))
            for i in range(nr_t_fits)
        ]

        t_fits = [
            t_fit for t_fit in t_fits if t_fit >= event.begin and t_fit <= event.end
        ]

        print(f"Multiple fitting points set to {t_fits}")
        print(f"Event has been observed entirely, duration: {event_duration} hours")

    else:
        nr_t_fits = 5
        event_duration = (event.end - event.begin).total_seconds() / 3600

        difference = event_duration / (nr_t_fits + 1)

        t_fits = [
            event.begin + datetime.timedelta(hours=difference * (i + 1))
            for i in range(nr_t_fits)
        ]

        # round t_fits to 30 minutes
        t_fits = [
            t_fit
            - datetime.timedelta(
                minutes=t_fit.minute % 30,
                seconds=t_fit.second,
                microseconds=t_fit.microsecond,
            )
            for t_fit in t_fits
        ]

        # remove t_fits that are outside the event
        t_fits = [
            t_fit for t_fit in t_fits if t_fit >= event.begin and t_fit <= event.end
        ]

        print(f"Multiple fitting points set to {t_fits}")
        print(f"Event has been observed entirely, duration: {event_duration} hours")

        t_s = event.begin
        t_e = event.end

    t_launch = elevo_event.launch_time  # begin - datetime.timedelta(hours=48)

    ############################
    #### Plot fitting points ###
    ############################

    plot_t_data = [t_point for t_point in t_data if t_point <= event.end]
    plot_b_data = b_data[: len(plot_t_data)]

    fig, ax = plot_insitu(plot_t_data, plot_b_data)

    ax.set_title(f"Fitting points for Event {elevo_event.event_id}")

    ax.axvline(x=t_s, lw=lw_fitp, alpha=0.75, color="k", ls="-.")
    ax.axvline(x=t_e, lw=lw_fitp, alpha=0.75, color="k", ls="-.")

    for _ in t_fits:
        ax.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")

    fig.tight_layout()
    fig.show()

    ############################
    #### Initializing method ###
    ############################

    base_fitter = BaseMethod()
    base_fitter.initialize(
        dt_0=t_launch,
        model=ToroidalModel,
        model_kwargs=model_kwargs,
    )
    base_fitter.add_observer(
        observer=sc,
        dt=t_fits,
        dt_s=t_s,
        dt_e=t_e,
    )

    t_launch = sanitize_dt(t_launch)

    if multiprocessing == True:

        # global mpool
        mpool = mp.Pool(processes=njobs)  # initialize Pool for multiprocessing
        processes.append(mpool)

    ##################################
    #### Initializing fitting data ###
    ##################################

    data_obj = FittingData(
        base_fitter.observers,
        fit_coord_system,
        b_data=b_data,
        t_data=t_data,
        pos_data=pos_data,
    )
    data_obj.generate_noise("psd", 30)

    ##################################
    #### Running the fitting method ##
    ##################################

    kill_flag = False
    pcount = 0
    timer_iter = None

    extra_args = {}

    try:
        for iter_i in range(iter_i, itermax):
            # We first check if the minimum number of iterations has been reached. If yes, we check if the target value for epsilon "epsgoal" has been reached.

            reached = False

            if iter_i >= itermin:
                if hist_eps[-1] < epsgoal:
                    print("Fitting terminated, target RMSE reached: eps < ", epsgoal)
                    kill_flag = True
                    break

            print("Minutes passed: ", time.gmtime(np.sum(hist_time)).tm_min)
            if time.gmtime(np.sum(hist_time)).tm_min > time_limit:
                print("Fitting terminated, time limit reached")
                kill_flag = True
                break

            print("Running iteration " + str(iter_i))

            timer_iter = time.time()

            # correct observer arrival times

            if iter_i >= len(time_offsets):
                _time_offset = time_offsets[-1]
            else:
                _time_offset = time_offsets[iter_i]

            data_obj.generate_data(_time_offset)
            # print(data_obj.data_b)
            # print(data_obj.data_dt)
            # print(data_obj.data_o)
            # print('success datagen')

            print(f"Time points: {data_obj.data_dt}")

            if len(hist_eps) == 0:
                eps_init = data_obj.sumstat(
                    [np.zeros((1, 3))] * len(data_obj.data_b), use_mask=False
                )[0]
                # returns summary statistic for a vector of zeroes for each observer
                hist_eps = [eps_init, eps_init * 0.98]
                # hist_eps gets set to the eps_init and 98% of it
                hist_eps_dim = len(eps_init)  # number of observers

                print("Initial eps_init = ", eps_init)

                model_obj_kwargs = dict(model_kwargs)
                model_obj_kwargs["ensemble_size"] = n_particles
                model_obj = base_fitter.model(
                    t_launch, **model_obj_kwargs
                )  # model gets initialized
            sub_iter_i = 0  # keeps track of subprocesses

            _random_seed = (
                random_seed + 100000 * iter_i
            )  # set random seed to ensure reproducible results
            # worker_args get stored

            worker_args = (
                iter_i,
                t_launch,
                base_fitter.model,
                model_kwargs,
                model_obj.iparams_arr,
                model_obj.iparams_weight,
                model_obj.iparams_kernel_decomp,
                data_obj,
                summary_type,
                hist_eps[-1],
                kernel_mode,
            )

            print("Starting simulations")

            if multiprocessing == True:
                print("Multiprocessing is used")
                _results = mpool.starmap(
                    abc_smc_worker,
                    [(*worker_args, _random_seed + i) for i in range(njobs)],
                )  # starmap returns a function for all given arguments
            else:
                print("Multiprocessing is not used")
                _results = starmap(
                    abc_smc_worker,
                    [(*worker_args, _random_seed + i) for i in range(njobs)],
                )  # starmap returns a function for all given arguments

            # the total number of runs depends on the ensemble size set in the model kwargs and the number of jobs
            total_runs = njobs * int(model_kwargs["ensemble_size"])  #
            # repeat until enough samples are collected
            while True:
                pcounts = [
                    len(r[1]) for r in _results
                ]  # number of particles collected per job
                _pcount = sum(pcounts)  # number of particles collected in total
                dt_pcount = (
                    _pcount - pcount
                )  # number of particles collected in current iteration
                pcount = _pcount  # particle count gets updated

                # iparams and according errors get stored in array
                particles_temp = np.zeros(
                    (pcount, model_obj.iparams_arr.shape[1]), model_obj.dtype
                )
                epses_temp = np.zeros((pcount, hist_eps_dim), model_obj.dtype)
                for i in range(0, len(_results)):
                    particles_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[
                        i
                    ][
                        0
                    ]  # results of current iteration are stored
                    epses_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[i][
                        1
                    ]  # errors of current iteration are stored

                sys.stdout.flush()
                print(
                    f"Step {iter_i}:{sub_iter_i} with ({pcount}/{n_particles}) particles",
                    end="\r",
                )

                time_passed = time.time() - timer_iter

                if time.gmtime(np.sum(hist_time) + time_passed).tm_min > time_limit:
                    print("Fitting terminated during iteration, time limit reached")
                    kill_flag = True
                    break

                # Flush the output buffer to update the line immediately

                if pcount > n_particles:
                    print(str(pcount) + " reached particles                     ")
                    break
                # if ensemble size isn't reached, continue
                # random seed gets updated

                _random_seed = random_seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)

                if multiprocessing == True:
                    _results_ext = mpool.starmap(
                        abc_smc_worker,
                        [(*worker_args, _random_seed + i) for i in range(njobs)],
                    )  # starmap returns a function for all given arguments
                else:
                    _results_ext = starmap(
                        abc_smc_worker,
                        [(*worker_args, _random_seed + i) for i in range(njobs)],
                    )  # starmap returns a function for all given arguments

                _results.extend(_results_ext)  # results get appended to _results
                sub_iter_i += 1
                # keep track of total number of runs
                total_runs += njobs * int(model_kwargs["ensemble_size"])  #

                if pcount == 0:
                    print("No hits, aborting                ")
                    kill_flag = True
                    break

            if kill_flag:
                break

            if pcount > n_particles:  # no additional particles are kept
                particles_temp = particles_temp[:n_particles]

            # if we're in the first iteration, the weights and kernels have to be initialized. Otherwise, they're updated.
            if iter_i == 0:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=False,
                    kernel_mode=kernel_mode,
                )  # replace iparams_arr by particles_temp
                model_obj.iparams_weight = (
                    np.ones((n_particles,), dtype=model_obj.dtype) / n_particles
                )
                model_obj.update_kernels(kernel_mode=kernel_mode)
            else:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=True,
                    kernel_mode=kernel_mode,
                )
            if isinstance(eps_quantile, float):
                new_eps = np.quantile(epses_temp, eps_quantile, axis=0)
                # after the first couple of iterations, the new eps gets simply set to the its maximum value instead of choosing a different eps for each observer

                if balanced_iterations > iter_i:
                    new_eps[:] = np.max(new_eps)

                hist_eps.append(new_eps)

            elif isinstance(eps_quantile, list) or isinstance(eps_quantile, np.ndarray):
                eps_quantile_eff = eps_quantile ** (1 / hist_eps_dim)  #
                _k = len(eps_quantile_eff)  #
                new_eps = np.array(
                    [
                        np.quantile(epses_temp, eps_quantile_eff[i], axis=0)[i]
                        for i in range(_k)
                    ]
                )
                hist_eps.append(new_eps)

            print(f"Setting new eps: {hist_eps[-2]} => {hist_eps[-1]}")

            hist_time.append(time.time() - timer_iter)

            print(
                f"Step {iter_i} done, {total_runs / 1e6:.2f}M runs in {time.time() - timer_iter:.2f} seconds, (total: {time.strftime('%Hh %Mm %Ss', time.gmtime(np.sum(hist_time)))})"
            )

            iter_i = iter_i + 1  # iter_i gets updated

            extra_args = {
                "t_launch": t_launch,
                "model_kwargs": model_kwargs,
                "hist_eps": hist_eps,
                "hist_eps_dim": hist_eps_dim,
                "base_fitter": base_fitter,
                "model_obj": model_obj,
                "data_obj": data_obj,
                "epses": epses_temp,
            }
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError: {e}, fitting terminated")
        kill_flag = True

    finally:
        for process in processes:
            process.terminate()

    return extra_args


def run_standard_fit(event: Event, df: pd.DataFrame) -> None:

    template = "none"
    bg_color = "rgba(0, 0,0, 0)"
    line_color = "black"
    line_colors = ["#c20078", "#f97306", "#069af3", "#000000"]
    eventshade = "LightSalmon"

    lw_insitu = 2  # linewidth for plotting the in situ data
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    lw_fitp = 2  # linewidth for plotting the lines where fitting points

    event_data = event.get_data(df, delta=6)

    core_data = CoreData(
        observer="RTSW",
        reference_frame="GSM",
        df=event_data,
    )

    t_data, b_data, pos_data = core_data.get_data("HEEQ")

    observer = CustomObserver(observer=core_data.spacecraft)

    modelstatevar_ranges = [int(2**16)] + [  # ensemblesize
        [-50, 50],  # Longitude (HEEQ)
        [-90, 90],  # Latitude (HEEQ)
        [-180.0, 180.0],  # Inclination
        [0.06, 0.3],  # Diameter 1 AU
        [1.0, 3.5],  # Aspect Ratio
        [21.5, 21.5],  # Launch Radius
        [400, 1500],  # Launch Velocity
        [1.0, 2.0],  # Expansion Rate
        [0.20, 2.00],  # Background Drag
        [100.0, 700.0],  # Background Velocity
        [-250.0, 250.0],  # T_Factor
        [1.0, 2.0],  # Magnetic Decay Rate
        [5.0, 50.0],  # Magnetic Field Strength 1 AU
    ]

    model_kwargs = get_modelkwargs_ranges(modelstatevar_ranges)
    iter_i = 0  # keeps track of iterations
    hist_eps = []  # keeps track of epsilon values
    hist_time = []  # keeps track of time

    balanced_iterations = 3
    time_offsets = [0]
    eps_quantile = 0.25
    epsgoal = 0.25
    kernel_mode = "cm"
    random_seed = 42
    summary_type = "norm_rmse"
    fit_coord_system = "HEEQ"
    sc = core_data.spacecraft

    multiprocessing = False
    njobs = 4

    itermin = 12
    itermax = 15

    n_particles = 512

    ############################
    #### Set fitting points ####
    ############################

    event_duration = (event.end - event.begin).total_seconds() / 3600
    nr_t_fits = min(event_duration - 1, 5)

    difference = event_duration / (nr_t_fits + 1)
    t_fits = [
        event.begin + datetime.timedelta(hours=difference * (i + 1))
        for i in range(nr_t_fits)
    ]

    # round t_fits to 30 minutes
    t_fits = [
        t_fit
        - datetime.timedelta(
            minutes=t_fit.minute % 30,
            seconds=t_fit.second,
            microseconds=t_fit.microsecond,
        )
        for t_fit in t_fits
    ]

    # remove t_fits that are outside the event
    t_fits = [t_fit for t_fit in t_fits if t_fit >= event.begin and t_fit <= event.end]

    print(f"Fitting event: {event.event_id}")
    print(f"Event begin: {event.begin}")
    print(f"Event end: {event.end}")
    print("Event duration: ", event_duration)
    print("Fitting points: ", t_fits)

    t_launch = event.begin - datetime.timedelta(hours=48)

    ############################
    #### Plot fitting points ###
    ############################

    fig, ax = plot_insitu(t_data, b_data)

    ax.set_title(f"Fitting points for Event {event.event_id}")

    ax.axvline(x=event.begin, lw=lw_fitp, alpha=0.75, color="k", ls="-.")
    ax.axvline(x=event.end, lw=lw_fitp, alpha=0.75, color="k", ls="-.")

    for _ in t_fits:
        ax.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")

    fig.tight_layout()
    fig.show()

    ############################
    #### Initializing method ###
    ############################

    base_fitter = BaseMethod()
    base_fitter.initialize(
        dt_0=t_launch,
        model=py3dcore.models.toroidal.ToroidalModel,
        model_kwargs=model_kwargs,
    )
    base_fitter.add_observer(
        observer=sc,
        dt=t_fits,
        dt_s=event.begin,
        dt_e=event.end,
    )

    t_launch = sanitize_dt(t_launch)

    if multiprocessing == True:

        # global mpool
        mpool = mp.Pool(processes=njobs)  # initialize Pool for multiprocessing
        processes.append(mpool)

    ##################################
    #### Initializing fitting data ###
    ##################################

    data_obj = FittingData(
        base_fitter.observers,
        fit_coord_system,
        b_data=b_data,
        t_data=t_data,
        pos_data=pos_data,
    )
    data_obj.generate_noise("psd", 30)

    ##################################
    #### Running the fitting method ##
    ##################################

    kill_flag = False
    pcount = 0
    timer_iter = None

    extra_args = {}

    try:
        for iter_i in range(iter_i, itermax):
            # We first check if the minimum number of iterations has been reached. If yes, we check if the target value for epsilon "epsgoal" has been reached.

            reached = False

            if iter_i >= itermin:
                if hist_eps[-1] < epsgoal:
                    print("Fitting terminated, target RMSE reached: eps < ", epsgoal)
                    kill_flag = True
                    break

            print("Minutes passed: ", time.gmtime(np.sum(hist_time)).tm_min)
            if time.gmtime(np.sum(hist_time)).tm_min > 12:
                print("Fitting terminated, time limit reached")
                kill_flag = True
                break

            print("Running iteration " + str(iter_i))

            timer_iter = time.time()

            # correct observer arrival times

            if iter_i >= len(time_offsets):
                _time_offset = time_offsets[-1]
            else:
                _time_offset = time_offsets[iter_i]

            data_obj.generate_data(_time_offset)
            # print(data_obj.data_b)
            # print(data_obj.data_dt)
            # print(data_obj.data_o)
            # print('success datagen')

            print(f"Time points: {data_obj.data_dt}")

            if len(hist_eps) == 0:
                eps_init = data_obj.sumstat(
                    [np.zeros((1, 3))] * len(data_obj.data_b), use_mask=False
                )[0]
                # returns summary statistic for a vector of zeroes for each observer
                hist_eps = [eps_init, eps_init * 0.98]
                # hist_eps gets set to the eps_init and 98% of it
                hist_eps_dim = len(eps_init)  # number of observers

                print("Initial eps_init = ", eps_init)

                model_obj_kwargs = dict(model_kwargs)
                model_obj_kwargs["ensemble_size"] = n_particles
                model_obj = base_fitter.model(
                    t_launch, **model_obj_kwargs
                )  # model gets initialized
            sub_iter_i = 0  # keeps track of subprocesses

            _random_seed = (
                random_seed + 100000 * iter_i
            )  # set random seed to ensure reproducible results
            # worker_args get stored

            worker_args = (
                iter_i,
                t_launch,
                base_fitter.model,
                model_kwargs,
                model_obj.iparams_arr,
                model_obj.iparams_weight,
                model_obj.iparams_kernel_decomp,
                data_obj,
                summary_type,
                hist_eps[-1],
                kernel_mode,
            )

            print("Starting simulations")

            if multiprocessing == True:
                print("Multiprocessing is used")
                _results = mpool.starmap(
                    abc_smc_worker,
                    [(*worker_args, _random_seed + i) for i in range(njobs)],
                )  # starmap returns a function for all given arguments
            else:
                print("Multiprocessing is not used")
                _results = starmap(
                    abc_smc_worker,
                    [(*worker_args, _random_seed + i) for i in range(njobs)],
                )  # starmap returns a function for all given arguments

            # the total number of runs depends on the ensemble size set in the model kwargs and the number of jobs
            total_runs = njobs * int(model_kwargs["ensemble_size"])  #
            # repeat until enough samples are collected
            while True:
                pcounts = [
                    len(r[1]) for r in _results
                ]  # number of particles collected per job
                _pcount = sum(pcounts)  # number of particles collected in total
                dt_pcount = (
                    _pcount - pcount
                )  # number of particles collected in current iteration
                pcount = _pcount  # particle count gets updated

                # iparams and according errors get stored in array
                particles_temp = np.zeros(
                    (pcount, model_obj.iparams_arr.shape[1]), model_obj.dtype
                )
                epses_temp = np.zeros((pcount, hist_eps_dim), model_obj.dtype)
                for i in range(0, len(_results)):
                    particles_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[
                        i
                    ][
                        0
                    ]  # results of current iteration are stored
                    epses_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[i][
                        1
                    ]  # errors of current iteration are stored

                sys.stdout.flush()
                print(
                    f"Step {iter_i}:{sub_iter_i} with ({pcount}/{n_particles}) particles",
                    end="\r",
                )
                # Flush the output buffer to update the line immediately

                if pcount > n_particles:
                    print(str(pcount) + " reached particles                     ")
                    break
                # if ensemble size isn't reached, continue
                # random seed gets updated

                _random_seed = random_seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)

                if multiprocessing == True:
                    _results_ext = mpool.starmap(
                        abc_smc_worker,
                        [(*worker_args, _random_seed + i) for i in range(njobs)],
                    )  # starmap returns a function for all given arguments
                else:
                    _results_ext = starmap(
                        abc_smc_worker,
                        [(*worker_args, _random_seed + i) for i in range(njobs)],
                    )  # starmap returns a function for all given arguments

                _results.extend(_results_ext)  # results get appended to _results
                sub_iter_i += 1
                # keep track of total number of runs
                total_runs += njobs * int(model_kwargs["ensemble_size"])  #

                if pcount == 0:
                    print("No hits, aborting                ")
                    kill_flag = True
                    break

            if kill_flag:
                break

            if pcount > n_particles:  # no additional particles are kept
                particles_temp = particles_temp[:n_particles]

            # if we're in the first iteration, the weights and kernels have to be initialized. Otherwise, they're updated.
            if iter_i == 0:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=False,
                    kernel_mode=kernel_mode,
                )  # replace iparams_arr by particles_temp
                model_obj.iparams_weight = (
                    np.ones((n_particles,), dtype=model_obj.dtype) / n_particles
                )
                model_obj.update_kernels(kernel_mode=kernel_mode)
            else:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=True,
                    kernel_mode=kernel_mode,
                )
            if isinstance(eps_quantile, float):
                new_eps = np.quantile(epses_temp, eps_quantile, axis=0)
                # after the first couple of iterations, the new eps gets simply set to the its maximum value instead of choosing a different eps for each observer

                if balanced_iterations > iter_i:
                    new_eps[:] = np.max(new_eps)

                hist_eps.append(new_eps)

            elif isinstance(eps_quantile, list) or isinstance(eps_quantile, np.ndarray):
                eps_quantile_eff = eps_quantile ** (1 / hist_eps_dim)  #
                _k = len(eps_quantile_eff)  #
                new_eps = np.array(
                    [
                        np.quantile(epses_temp, eps_quantile_eff[i], axis=0)[i]
                        for i in range(_k)
                    ]
                )
                hist_eps.append(new_eps)

            print(f"Setting new eps: {hist_eps[-2]} => {hist_eps[-1]}")

            hist_time.append(time.time() - timer_iter)

            print(
                f"Step {iter_i} done, {total_runs / 1e6:.2f}M runs in {time.time() - timer_iter:.2f} seconds, (total: {time.strftime('%Hh %Mm %Ss', time.gmtime(np.sum(hist_time)))})"
            )

            iter_i = iter_i + 1  # iter_i gets updated

            extra_args = {
                "t_launch": t_launch,
                "model_kwargs": model_kwargs,
                "hist_eps": hist_eps,
                "hist_eps_dim": hist_eps_dim,
                "base_fitter": base_fitter,
                "model_obj": model_obj,
                "data_obj": data_obj,
                "epses": epses_temp,
            }
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError: {e}, fitting terminated")
        kill_flag = True

    finally:
        for process in processes:
            process.terminate()

    return extra_args
