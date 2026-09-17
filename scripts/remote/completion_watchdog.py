#!/usr/bin/env python3
"""Laptop-side, completion-aware cost cap for a remote run.

WHY THIS EXISTS
---------------
The pod-side cost safety net (the sentinel self-stop and the --max-hours
timer) both authenticate with RUNPOD_STOP_KEY. When that key cannot stop
pods -- provision prints "the pod-side stop key CANNOT stop pods on either
API surface" -- NEITHER fires and the run has no automatic cap at all.

A naive laptop-side watchdog that only terminates at a deadline is a poor
substitute: it cannot tell that a run FINISHED, so the pod idles at full GPU
rate until the deadline. That cost $15.61 on pawlowski_kineform_h3_v2
(training done ~13.6h in, watchdog fired at 17h).

This watchdog polls for the run reaching a terminal state and then tears
down properly, with the deadline kept only as a backstop.

WHAT IT DOES
------------
  * polls `cli.py watch --once --json` for a terminal exit code
  * on terminal: waits --grace-minutes (so an interactive session can finish
    pulling/reviewing), then runs `cli.py down <run>` -- a VERIFIED final
    pull + terminate, not a raw terminate
  * on deadline: same `down`, so artifacts are still pulled and verified
  * `down` refuses to terminate when verification fails and STOPS the pod
    instead, preserving the volume for `rescue`; that refusal is logged
    loudly rather than forced

LIMITS -- still a backstop, not a guarantee. It dies with a laptop shutdown
or hard sleep, which is exactly the case the pod-side stop covers. The real
fix is granting RUNPOD_STOP_API_KEY api.runpod.io/graphql Read/Write in the
RunPod console.

Usage:
    nohup python3 scripts/remote/completion_watchdog.py <run> \
        --max-hours 17 --grace-minutes 30 > /dev/null 2>&1 &

    # verify the wiring without arming anything
    python3 scripts/remote/completion_watchdog.py <run> --dry-run
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# exit codes from `watch --once --json` that mean "training is over"
TERMINAL = {0: "COMPLETED", 20: "CRASHED", 21: "STOPPED",
            22: "TIMED_OUT", 24: "POD_LOST"}


def log(path, msg):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", buffering=1) as fh:
            fh.write(line + "\n")


def poll(run):
    """-> (exit_code, state, step) ; None on a transient failure."""
    try:
        p = subprocess.run(
            [sys.executable, "scripts/remote/cli.py", "watch", run, "--once", "--json"],
            capture_output=True, text=True, timeout=300, cwd=REPO)
        d = json.loads(p.stdout.strip().splitlines()[-1])
        return d.get("exit_code"), d.get("state"), d.get("step")
    except Exception:
        return None


def teardown(run, logpath, reason):
    log(logpath, f"{reason} -> running verified `down {run}`")
    p = subprocess.run([sys.executable, "scripts/remote/cli.py", "down", run],
                       capture_output=True, text=True, cwd=REPO)
    for ln in (p.stdout or "").splitlines()[-6:]:
        log(logpath, f"  {ln}")
    if p.returncode != 0:
        log(logpath, f"  down exited {p.returncode} — artifacts may be missing. "
                     f"Pod is STOPPED, not terminated: volume still bills. "
                     f"Run `rescue {run}` then verify. MANUAL ACTION REQUIRED.")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--max-hours", type=float, default=17.0,
                    help="hard backstop; tears down even if state never goes terminal")
    ap.add_argument("--grace-minutes", type=float, default=30.0,
                    help="wait after a terminal state before tearing down")
    ap.add_argument("--poll-seconds", type=float, default=300.0)
    ap.add_argument("--log", default=os.path.expanduser(
        "~/.cache/aitk-watchdog/completion_watchdog.log"))
    ap.add_argument("--dry-run", action="store_true",
                    help="poll once, report what WOULD happen, then exit")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    mpath = os.path.join(REPO, "runs", args.run, "manifest.json")
    if not os.path.exists(mpath):
        print(f"no manifest for run {args.run}", file=sys.stderr); sys.exit(1)
    pod = json.load(open(mpath)).get("pod_id")

    if args.dry_run:
        r = poll(args.run)
        print(f"run={args.run} pod={pod}")
        print(f"poll -> {r}")
        if r and r[0] in TERMINAL:
            print(f"WOULD wait {args.grace_minutes}min then `down {args.run}` "
                  f"(terminal: {TERMINAL[r[0]]})")
        elif r:
            print(f"WOULD keep polling every {args.poll_seconds:.0f}s; "
                  f"backstop at {args.max_hours}h")
        else:
            print("poll failed (transient?) — would retry")
        return

    deadline = time.time() + args.max_hours * 3600
    log(args.log, f"armed run={args.run} pod={pod} backstop={args.max_hours}h "
                  f"grace={args.grace_minutes}min")
    while time.time() < deadline:
        r = poll(args.run)
        if r is None:
            log(args.log, "poll failed (transient) — retrying"); time.sleep(args.poll_seconds); continue
        ec, state, step = r
        if ec in TERMINAL:
            log(args.log, f"terminal: {TERMINAL[ec]} (exit {ec}) at step {step}; "
                          f"waiting {args.grace_minutes}min grace")
            time.sleep(args.grace_minutes * 60)
            sys.exit(teardown(args.run, args.log, f"{TERMINAL[ec]} + grace elapsed"))
        time.sleep(args.poll_seconds)
    sys.exit(teardown(args.run, args.log,
                      f"BACKSTOP {args.max_hours}h reached (state never went terminal)"))


if __name__ == "__main__":
    main()
