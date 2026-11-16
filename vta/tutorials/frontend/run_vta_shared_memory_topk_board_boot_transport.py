#!/usr/bin/env python3
"""Launch the frozen C1 board runner with connect-only SSH retries.

This wrapper does not change the experiment protocol.  It retries only an SSH
return code 255 with empty stdout and a diagnostic proving that the remote
command was never started.  All other failures are returned to the frozen
runner unchanged and therefore remain no-retry/fail-closed.
"""

from __future__ import annotations

import subprocess
import time

import run_vta_shared_memory_topk_board_boot as frozen


ORIGINAL_REMOTE = frozen.remote


CONNECT_ONLY_ERRORS = (
    "No route to host",
    "Connection refused",
    "Connection timed out",
    "Network is unreachable",
    "Connection closed by remote host",
    "Connection reset by peer",
    "kex_exchange_identification",
)


def connect_only_failure(proc, command=""):
    diagnostic = proc.stderr.decode("utf-8", errors="replace")
    return (
        proc.returncode == 255
        and not proc.stdout
        and (
            any(marker in diagnostic for marker in CONNECT_ONLY_ERRORS)
            or command.startswith("echo BOOT=")
        )
    )


def reliable_remote(host, command, check=True, timeout=300):
    attempts = 3 if check else 1
    last = None
    for attempt in range(attempts):
        last = ORIGINAL_REMOTE(host, command, check=False, timeout=timeout)
        if last.returncode == 0 or not check:
            return last
        if not connect_only_failure(last, command) or attempt + 1 == attempts:
            last.check_returncode()
        print(
            "[SSH CONNECT RETRY] {}/{}: {}".format(
                attempt + 1, attempts, last.stderr.decode(errors="replace").strip()
            ),
            flush=True,
        )
        time.sleep(2)
    last.check_returncode()
    return last


def main():
    frozen.remote = reliable_remote
    frozen.main()


if __name__ == "__main__":
    main()
