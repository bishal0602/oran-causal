# Provenance: experiments launch (Kaggle, Colab, VPS, dispatcher)

Source commit `CDD-ORAN@0c89106`. Ported code keeps behaviour; port_map §2.4 (cloud rows). Lightning (`xm_lightning.py`, `lightning_run.py`, the `lightning` branch of `campaign.main`) is dropped (drop-historical). Every path below is under `packages/experiments/src/oran_causal/experiments/launch/` unless it starts with `packages/experiments/tests/`.

| New path | Source | Note |
|---|---|---|
| `bundle.py` | `scratchpad/e6_dev/cloud.py` (bundle, MANIFEST, `sha`), `kaggle_job.py` (`_files`, `build`, dot-encoding), `vps_run.py` (`bundle`), `cdd_oran/xmethod/campaign.py` (`_git_blob`, frozen protocol copy) | One bundler for the three platforms. Base set: `packages/*/src`, member and root `pyproject.toml`, `uv.lock`, `.python-version`, protocols, manifests, artifacts (the old one needed `--paths` for the data). CRLF → LF for text; binaries untouched. `MANIFEST.json` has git head, dirty, per-file sha256, `local_env`, `local_fp`. No `PIN=match` install: the lock is the pin |
| `fingerprint.py` | `cloud.py` (`numeric_env`, `fingerprint`, CLI `fingerprint`) | Same hashes (Generator methods, ufuncs, scipy, per-second plant trajectory); the plant comes from `oran_causal.sim.e6` |
| `commands.py` | `cdd_oran/xmethod/campaign.py` (`_cloud_cmd`, `_setup_cmd`, `launch_checks`) | `PartsPlan`, `SetupPlan` dataclasses instead of 10-argument functions. `XM_*` → `ORAN_CAUSAL_*`; log prefix `[xm-c]` kept. Setup is `uv sync --frozen` of the bundled lock plus the pin check, not `pip install --require-hashes -r requirements.cloud.txt`. Not ported: `write_lock`, `_req_blocks`, `lock_groups_and_indexes` (the lock is `uv.lock`, shipped as is) |
| `hosts.py` | `scratchpad/e6_dev/vps_run.py` (`MAX_PROCS`, `MemoryMax`, load refusal), `kaggle_job.py`/`colab_run.py` (wall defaults) | Caps as data: `HostProfile`, `LoadGuard`, `VPS_SHARED` (7 procs, 11G, swap 0, MemAvailable ≥ 12 GiB, other load ≤ 1, one job), `VPS_DEDICATED`, `KAGGLE_CPU`, `COLAB_CPU` |
| `runner.py` | new | Injectable process runner, clock and directories; replaces the module globals `ROOT`, `HERE`, `KAGGLE` |
| `kaggle.py` | `kaggle_run.py`, `kaggle_job.py` | `KaggleJob`, `stage`, `KaggleClient`, `launch`, `pull`. The e6dev shard kernels of `kaggle_run.py` (`SHARDS`, `grid.py`) are not ported. `PIN` is dropped |
| `vps.py` | `vps_run.py` | `Transport` protocol (`Ssh` or a fake), `Vps(transport, profile)`: probe, setup, push, venv, launch, status, pull, stop. One venv per job (workspace members are editable and point into the job's bundle). Base `/opt/oran-causal`, unit prefix `oran-causal-` (was `/opt/cdd-xm`, `cdd-xm-`) |
| `colab_session.py` | `colab_run.py` (CLI wrapper, `JOB_*` remote templates, `new`, `stop`, `gone`, `wait_gone`, session probe) | `Colab(invoke, probe, clock, accelerator)`. `WORK` = `/content/oran-causal-run` |
| `colab.py` | `colab_run.py` (`job`, `_job_tick`, `_job_finish`, `_job_lost`, `tick.lock`) | Tick state machine as functions of `ColabEnv`. Dropped: parts mode, resume, smoke, probe, timing sub-commands |
| `schedule.py` | `colab_run.py`, `xm_dispatch.py` (schtasks) | `Scheduler` protocol, `TaskScheduler`, `cron_line` (Linux; not in the old code) |
| `plan.py` | `campaign.py` `main` (kaggle, colab, vps branches) | `LaunchRequest` → `SessionPlan` → `execute`; `describe` is the dry run |
| `dispatch.py` | `xm_dispatch.py` | `Dispatcher` over a `Backend` protocol; same lanes, chunks (Kaggle 8, Colab 4, VPS 7), limits (Kaggle 5, own 4, Colab 2), back-offs and 3-attempt requeue. Lightning lane removed |
| `live.py` | `xm_dispatch.py` (real Kaggle/VPS calls) | `LiveBackend`, `default_ssh_vps` |
| `__main__.py` | `campaign.py` `main` (launch), `colab_run.py`, `kaggle_run.py`, `vps_run.py`, `xm_dispatch.py` CLIs | `python -m oran_causal.experiments.launch <launch\|colab\|kaggle\|vps\|dispatch\|fingerprint>` |
| `packages/experiments/tests/test_launch_commands.py` | `tests/test_xmethod_campaign.py` | Part env names, GPU commands, setup, EVAL one-per-CPU, session exit status (bash), VPS scope quota, profiles, EVAL launch checks |
| `.../tests/test_launch_bundle.py` | `tests/test_xmethod_campaign.py` (lock export, bundle imports every method, v4 docs bundled) | Base set, CRLF, MANIFEST, frozen text, dot-encoding; `test_a_bundle_imports_every_arm_without_the_workspace` is `slow` |
| `.../tests/test_launch_kaggle.py`, `test_launch_vps.py`, `test_launch_colab.py`, `test_launch_plan.py` | new (the cloud code had no tests) | Fake runner, transport, `colab` CLI, scheduler and backend; no network |

Old launch tests (`tests/test_xmethod_campaign.py`) and where they went:

| Old test | New |
|---|---|
| `test_install_lock_drops_only_the_torch_build` | dropped: no `uv export` lock; `test_the_base_set_is_exactly_what_a_run_needs` checks `uv.lock` and `.python-version` ship |
| `test_eval_python_312_venv_setup` | `test_launch_commands.py::test_setup_installs_the_bundled_lock_and_checks_the_pins`, `test_launch_vps.py::test_the_venv_step_syncs_the_lock_and_checks_the_interpreter` |
| `test_gpu_cloud_and_setup_commands` | `test_launch_commands.py::test_gpu_cloud_and_setup_commands` |
| `test_cloud_bundle_imports_every_method` | `test_launch_bundle.py::test_a_bundle_imports_every_arm_without_the_workspace` (`slow`) |
| `test_lightning_and_colab_gpu_launch_commands` | Colab part: `test_launch_plan.py::test_describe_lists_the_steps_of_each_platform` (gpu); Lightning part dropped |
| `test_r55_eval_cloud_cmd_caps_processes_at_quota` | `test_launch_commands.py::test_eval_cloud_cmd_caps_processes_at_the_cpu_quota` |
| `test_vps_launch_steps` | `test_launch_vps.py::test_push_then_launch_records_both`, `test_the_launch_script_runs_in_a_capped_scope` |
| `test_r59_f5_session_exit_nonzero_if_any_part_fails` | `test_launch_commands.py::test_session_exit_is_nonzero_if_any_part_fails` |
| `test_r59_f8_vps_scope_cpuquota_follows_procs` | `test_launch_commands.py::test_the_shared_production_profile_holds_the_old_caps`, `test_launch_vps.py::test_the_launch_script_runs_in_a_capped_scope` |
| `test_eval_run_refused_before_any_data` (launch half) | `test_launch_commands.py::test_eval_launch_needs_an_authorised_protocol_and_a_clean_tree` |
