import argparse
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def update_input_paths(config_path, base_dir):
    with open(config_path) as f:
        cfg = json.load(f)
    jt = cfg.get("JOB_TYPE", "").strip()
    js = cfg.setdefault("JOB_SPECIFIC_SETTINGS", {}).setdefault(jt, {})
    for k, v in js.items():
        if k.endswith("_TERMS_FILE") and isinstance(v, str) and not os.path.isabs(v):
            js[k] = os.path.join(base_dir, v)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)

def get_job_type(config_path):
    with open(config_path) as f:
        return json.load(f).get("JOB_TYPE", "unknown").strip()

def setup_logger(parent_dir, job_type):
    logger = logging.getLogger("SKiM-GPT-wrapper")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = "%(asctime)s - SKiM-GPT-wrapper - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(formatter); logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(parent_dir, f"{job_type}_wrapper.log"))
    fh.setFormatter(formatter); logger.addHandler(fh)
    return logger

def update_censor_year(config_path, year, depth):
    data = json.load(open(config_path))
    job_type = data.setdefault("JOB_TYPE", "").strip()
    job_specific_settings = data.setdefault("JOB_SPECIFIC_SETTINGS", {}).setdefault(job_type, {})
    #job_specific_settings[job_type]["censor_year"] = year
    # add censor_year_upper and censor_year_lower
    job_specific_settings["censor_year_upper"] = year
    if depth == 1:
        job_specific_settings["censor_year_lower"] = year
    else:
        depth = depth-1
        job_specific_settings["censor_year_lower"] = year-depth
    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)

def copy_project_src(src_root, dst_root):
    s = os.path.join(src_root, "src")
    d = os.path.join(dst_root, "src")
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        raise FileNotFoundError(f"{s} not found")

def parse_job_status(log_dir):
    out_base = os.path.join(log_dir, "output")
    if not os.path.isdir(out_base):
        return None
    subs = [d for d in os.listdir(out_base) if d.startswith("output_")]
    if not subs:
        return None
    logf = os.path.join(out_base, subs[0], "SKiM-GPT.log")
    if not os.path.isfile(logf):
        return None
    last = None
    for L in open(logf):
        if "status:" in L:
            last = L.strip()
    if not last:
        return None
    return re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ", "", last)

def run_one_year(year, work_dir, project_dir, original_config, main_py_path, logger, depth):
    logger.info(f"Starting censor_year {year}")
    cfg_path = os.path.join(work_dir, "config.json")
    shutil.copy2(original_config, cfg_path)

    update_input_paths(cfg_path, project_dir)
    update_censor_year(cfg_path, year, depth)
    copy_project_src(project_dir, work_dir)

    env = os.environ.copy()
    start = time.time()
    res = subprocess.run([sys.executable, main_py_path], cwd=work_dir, env=env)
    elapsed = time.time() - start

    if res.returncode != 0:
        logger.error(f"censor_year {year} failed (rc={res.returncode})")
    else:
        logger.info(f"censor_year {year} completed in {elapsed:.2f} seconds")
    return work_dir, res.returncode

def flatten_and_cleanup(parent_dir):
    out_root = os.path.join(parent_dir, "output")
    for cy in os.listdir(out_root):
        cy_path = os.path.join(out_root, cy)
        if not os.path.isdir(cy_path):
            continue
        inner = os.path.join(cy_path, "output")
        if os.path.isdir(inner):
            subs = [d for d in os.listdir(inner) if os.path.isdir(os.path.join(inner, d))]
            if subs:
                real = os.path.join(inner, subs[0])
                for item in os.listdir(real):
                    src = os.path.join(real, item)
                    dst = os.path.join(cy_path, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)
            shutil.rmtree(inner)
        for junk in ("src", "token", "secrets.json"):
            p = os.path.join(cy_path, junk)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)

def main():
    p = argparse.ArgumentParser(
        prog="main_wrapper.py",
        description="Run one main.py per censor-year in parallel."
    )
    p.add_argument("-censor_year_range", required=True, help="e.g. 1980-2000")
    p.add_argument("-censor_year_increment", type=int, required=True)
    p.add_argument("-censor_year_depth", type=int, required=False, default=1)
    args = p.parse_args()

    try:
        lo, hi = map(int, args.censor_year_range.split("-"))
    except:
        sys.exit("Invalid -censor_year_range, must be like 1980-2000")
    years = list(range(lo, hi+1, args.censor_year_increment))
    num_years = len(years)
    depth = int(args.censor_year_depth)
    #print(depth)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    master_cfg  = os.path.join(project_dir, "config.json")
    iters       = int(json.load(open(master_cfg))["GLOBAL_SETTINGS"].get("iterations", 1))
    ts          = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parent_name = f"output_{ts}_cy_range{lo}-{hi}_cy_inc_{args.censor_year_increment}_iterations{iters}"
    parent_dir  = os.path.join(os.path.abspath("output"), parent_name)
    os.makedirs(parent_dir, exist_ok=True)

    # --- Copy & fix up wrapper‐level config.json ---
    shutil.copy2(master_cfg, os.path.join(parent_dir, "config.json"))
    wrapper_cfg = os.path.join(parent_dir, "config.json")
    update_input_paths(wrapper_cfg, project_dir)
    copy_project_src(project_dir, parent_dir)

    # --- Logger ---
    jt     = get_job_type(wrapper_cfg)
    logger = setup_logger(parent_dir, jt)
    logger.info(f"Parent dir: {parent_dir}")
    logger.info(f"Preparing to run {num_years} years (each with {iters} iterations)")

    # export for children
    os.environ["WRAPPER_PARENT_DIR"]    = parent_dir
    os.environ["CENSOR_YEAR_RANGE"]     = args.censor_year_range
    os.environ["CENSOR_YEAR_INCREMENT"] = str(args.censor_year_increment)
    os.environ["CENSOR_YEAR_DEPTH"]     = str(args.censor_year_depth)

    main_py = os.path.join(project_dir, "main.py")

    # ── 1) Serial first‐year (for cost prompt) ───────────────────────────────
    first = years[0]
    first_dir = os.path.join(parent_dir, "output", f"output_{ts}_cy{first}")
    os.makedirs(first_dir, exist_ok=True)
    _, rc = run_one_year(first, first_dir, project_dir, wrapper_cfg, main_py, logger, depth)
    if rc != 0:
        logger.error("First-year run (with cost-prompt) failed; aborting wrapper")
        sys.exit(1)

    # Delete the cost‐only dir
    shutil.rmtree(first_dir, ignore_errors=True)
    logger.info(f"Removed cost-only directory for year {first}")

    # ── 2) Fire off all years in parallel ──────────────────────────────────
    work_dirs = { y: os.path.join(parent_dir, "output", f"output_{ts}_cy{y}") for y in years }
    for wd in work_dirs.values():
        os.makedirs(wd, exist_ok=True)

    with ThreadPoolExecutor(max_workers=len(work_dirs)) as exe:
        futures = {
            exe.submit(run_one_year, y, wd, project_dir, wrapper_cfg, main_py, logger, depth): y
            for y, wd in work_dirs.items()
        }

        while futures:
            for f in list(futures):
                if f.done():
                    y = futures.pop(f)
                    try:
                        _, rc = f.result()
                        if rc != 0:
                            logger.error(f"Year {y} failed")
                    except Exception as e:
                        logger.error(f"Year {y} exception: {e}")
            # poll statuses
            for y, wd in work_dirs.items():
                st = parse_job_status(wd)
                if st:
                    logger.info(f"censor_year {y} status: {st}")
            time.sleep(30)

    # ── 3) Merge + cleanup ───────────────────────────────────────────────────
    elapsed = time.time() - datetime.datetime.strptime(ts, "%Y%m%d%H%M%S").timestamp()
    logger.info(f"Wrapper completed in {elapsed:.2f} seconds")

    flatten_and_cleanup(parent_dir)
    logger.info("Per-year folders flattened and cleaned")

    merger = [ sys.executable,
               os.path.join(project_dir, "wrapper_result_merger.py"),
               "-parent_dir", parent_dir ]
    mr = subprocess.run(merger)
    if mr.returncode != 0:
        logger.error("Result merge failed")
    else:
        logger.info("Result merge succeeded")

    sentinel = os.path.join(parent_dir, ".cost_prompt_done")
    if os.path.isfile(sentinel):
        os.remove(sentinel)
        logger.info("Removed cost-prompt sentinel file")


if __name__ == "__main__":
    main()
