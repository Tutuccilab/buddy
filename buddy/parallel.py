"""
Helpers for processing several images at the same time (used by 01_fish_bulk_processing.ipynb).

The CPU-heavy steps of one image do not depend on any other image, so they can run side by side in
separate processes:
    read_one(...)    step 1: read the .vsi + DIC, write the TIFF layers
    spots_one(...)   step 3: focus, background filter, spot detection, decomposition, alignment
They run the functions in buddy/pipeline.py in a worker process.
GPU steps (Omnipose, bud model) stay in the notebook process, one image after the other.

Each worker records its step times in the image's img.json.

Workers are plain module-level functions (required on Windows, where new processes are started fresh).
Memory: one image needs a few GB while it is being filtered - choose the number of workers accordingly.
"""
import logging
import os
from pathlib import Path


def default_workers(gb_per_image=4):
    """A safe number of parallel workers: limited by CPU cores and by memory (gb_per_image per worker)."""
    cores = os.cpu_count() or 2
    try:
        import psutil
        memory_gb = psutil.virtual_memory().available / 1e9
        by_memory = max(1, int(memory_gb // gb_per_image))
    except ImportError:
        by_memory = 2
    return max(1, min(cores - 1, by_memory))


def _experiment(cfg_file, inputdir, outputdir, parameter_overrides=None):
    """The Experiment from the config file; parameter_overrides (dict) replaces entries of its "parameters" block."""
    from buddy.experiment import Experiment
    exp = Experiment.from_cfg_file(cfg_file)
    exp.inputdir, exp.outputdir = inputdir, outputdir
    if parameter_overrides:
        exp.parameters.update(parameter_overrides)
        exp.use_gpu = exp.parameters.get('use_gpu', True)
    return exp


def read_one(params, cfg_file, inputdir, outputdir, mode):
    """Step 1 for one image in a worker process (see buddy.pipeline.read_image)."""
    from buddy import pipeline
    exp = _experiment(cfg_file, inputdir, outputdir)
    return str(pipeline.read_image(params, exp, mode))


def spots_one(json_file, cfg_file, inputdir, outputdir, mode, parameter_overrides=None):
    """Step 3a for one image in a worker process (see buddy.pipeline.spots_image).
    parameter_overrides: e.g. {"gpu_spot_detection": False}."""
    from buddy import pipeline
    exp = _experiment(cfg_file, inputdir, outputdir, parameter_overrides)
    return str(pipeline.spots_image(json_file, exp, mode))


def iter_parallel(function, jobs, workers, should_stop=None):
    """
    Run function(*job) for every job in `workers` processes and yield (job, result, error) as each one
    finishes (error is None if it worked; one failing image does not stop the others).
    Only `workers` jobs run at a time; should_stop(): when it returns True, no new jobs are started
    (the running ones finish).
    """
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    waiting = list(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {}
        while waiting or running:
            while waiting and len(running) < workers and not (should_stop is not None and should_stop()):
                job = waiting.pop(0)
                running[pool.submit(function, *job)] = job
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                job = running.pop(future)
                try:
                    yield job, future.result(), None
                except Exception as e:
                    logging.error(f"image {job[0]} failed: {e!r}")
                    yield job, None, e


def run_parallel(function, jobs, workers):
    """Run function(*job) for every job in `workers` processes, printing progress; returns the results."""
    results = []
    for n, (job, result, error) in enumerate(iter_parallel(function, jobs, workers), start=1):
        if error is None:
            results.append(result)
            print(f"  {n}/{len(jobs)} done: {Path(str(result)).parent.name or result}", flush=True)
        else:
            print(f"  {n}/{len(jobs)} FAILED: {job[0]} ({error!r})", flush=True)
    return results
