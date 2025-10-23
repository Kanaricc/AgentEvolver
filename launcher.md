# Launcher Usage Guide (`launcher.py`)

This document explains the usage, arguments, and workflow of `launcher.py`, the main entry point for launching and managing experiments in the BeyondAgent project.

## Overview

`launcher.py` is a command-line tool designed to:
- Prepare experiment environments
- Backup configuration and code
- Optionally kill existing Python/Ray processes
- Launch various environment services (AppWorld, WebShop, BFCL, Exp Maker, LogView, Crafters)
- Start the main training or evaluation process

## Basic Usage


Run the launcher with desired options. Example:

```bash
python launcher.py --kill --conf examples/self-question-nav-attr.yaml --with-appworld --with-exp-maker --with-logview
```

This command will:
- Kill existing Python and Ray processes (except VSCode and itself)
- Use `examples/self-question-nav-attr.yaml` as the experiment configuration
- Launch the AppWorld and Exp Maker services
- Prepare backups and start the main experiment

## Command-Line Arguments

| Argument                | Type      | Description                                                                                 |
|------------------------|-----------|---------------------------------------------------------------------------------------------|
| `--target`             | str       | Target script/module to run (default: `beyondagent.main_ppo`)                               |
| `--conf`               | str       | Path to the experiment YAML configuration file                                               |
| `--db`                 | str       | Enable debug mode and set debug tags                                                         |
| `--with-appworld`      | flag      | Launch AppWorld environment service                                                          |
| `--with-webshop`       | flag      | Launch WebShop environment service                                                           |
| `--with-bfcl`          | flag      | Launch BFCL environment service                                                             |
| `--with-exp-maker`     | flag      | Launch Experience Maker service                                                              |
| `--with-logview`       | flag      | Launch LogView web service and open browser                                                  |
| `--with-crafters`      | flag      | Launch Crafters environment simulation                                                      |
| `-k`, `--kill`, `--python-killer` | flag | Kill existing Ray and Python processes before starting (recommended for clean start)         |

## Workflow Details

1. **Process Cleanup (Optional)**
   - If `--python-killer` is set, kills all Ray and Python processes except VSCode and itself.

2. **Configuration Handling**
   - Loads the YAML config file specified by `--conf`.
   - Determines experiment name from `trainer.experiment_name` or YAML filename.
   - Backs up code/config directories and the YAML file to `launcher_record/<exp_name>/backup/`.
   - Rewrites the YAML to set the correct experiment name and replace placeholders.

3. **Service Launching**
   - Launches selected environment services (AppWorld, WebShop, BFCL, Exp Maker, LogView, Crafters) as background processes using PTY.
   - For LogView, opens the web UI in a browser.

4. **Main Process Launch**
   - Runs the main training/evaluation script/module with the prepared config.
   - Passes environment variables for debugging and logging as needed.


## Persistent service management and logs

Services launched with flags like `--with-appworld`, `--with-exp-maker`, etc., are started via a companion process manager built on `beyondagent.utils.daemon.LaunchCommandWhenAbsent`.

What this gives you:

- Single-instance guarantee: if a service is already running, it won’t be launched again.
- Detached background execution: services keep running independently of your terminal session.
- Stable identifiers: each launch combination is hashed; its process group id (PGID) is stored next to the log.
- Structured logs: all companion logs live under `logs/companion/` with a file name pattern `<tag>.<hash>.<hostname>.log` and a sibling PGID file `<tag>.<hash>.<hostname>.pgid`.

On launch, the console prints where logs go, for example:

```
log to logs/companion/appworld_env_service.bc21d4e3.dlc1x89r0ps6ysm8-master-0.log
```

You can open or stream the service logs directly, e.g.:

```bash
tail -f logs/companion/appworld_env_service.*.log
```

Notes:

- Skips re-launch if an existing PGID is active. To restart, stop the service (kill the process group) or remove the corresponding `.pgid` file and relaunch the flag.
- For PTY-backed services, human-readable command execution is proxied via `beyondagent.utils.pty`; output still goes to the same log file.
- Tags: `launcher.py` applies a meaningful `tag` for each service (e.g., `appworld_env_service`), which appears in the log filename.


## Example: Full Experiment Launch

```bash
python launcher.py --kill --conf examples/self-question-nav-attr.yaml --with-appworld --with-exp-maker
```

- Cleans up old processes
- Backs up code and config
- Launches AppWorld and Exp Maker
- Starts the experiment with the specified YAML config

## Notes
- The backup directory is `launcher_record/<exp_name>/backup/`.
- The rewritten YAML is saved as `launcher_record/<exp_name>/yaml_backup.yaml`.
- Placeholders like `${trainer.experiment_name}` in the config are replaced automatically.
- For debugging, use `--db <tag>` to enable debug mode and set debug tags.
- To launch additional services, add their respective flags.

## Troubleshooting
- If you see errors about missing config or backup directories, check your YAML path and permissions.
- If services fail to start, ensure their paths/scripts are set in your environment variables or `.env` file.
- For port or resource conflicts, use `--kill` to clean up before launching.

---

For more details, see the comments in `launcher.py` or contact the project maintainers.

## Debugging with `--db`

Use `--db` to enable post-mortem debugging and set tag-based conditional breakpoints. The launcher will set:

- `RAY_DEBUG_POST_MORTEM=1`
- `DEBUG_TAGS=<value of --db>` (use `|` to separate multiple tags)
- `RAY_record_task_actor_creation_sites=true`

Then, inside your code, import the helper from `vsdb.py`:

```python
from vsdb import bp  # or: from vsdb import vscode_conditional_breakpoint as bp

def some_function():
   bp("tag")  # hits only if "tag" is in DEBUG_TAGS and RAY_DEBUG_POST_MORTEM is set
   # ... your logic ...
```

Run with one or more tags:

```bash
python launcher.py --kill --conf examples/self-question-nav-attr.yaml --db "tag|tag2|tag3|tag4"
```

Behavior summary:

- `bp()` with no arguments: triggers once when `RAY_DEBUG_POST_MORTEM` is set.
- `bp("tag")`: triggers (once by default) only if `tag` appears in `DEBUG_TAGS` (split by `|`).
- You can call `vscode_conditional_breakpoint(tag, once=False)` if you need to break every time.

Tip: The approach works well with the Ray Distributed Debugger VSCode extension. See the inline guide in `vsdb.py` for setup screenshots and more details.
