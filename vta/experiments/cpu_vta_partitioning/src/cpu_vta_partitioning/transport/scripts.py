"""POSIX shell fragments shared by the live transport and local shell tests."""
import shlex


def session_lock(base, boot, identity, failure="exit 95"):
    q = shlex.quote
    parent = base.rstrip("/") + "/locks/" + boot
    lock = parent + "/session.lock"
    return [
        'test "$(cat /proc/sys/kernel/random/boot_id)" = ' + q(boot) + " || exit 99",
        "mkdir -p " + q(parent) + " || exit 97",
        "mkdir " + q(lock) + " 2>/dev/null || { " + failure + "; }",
        # A new boot uses a different path. Never reclaim a same-boot lock by PID
        # alone: a surviving child may still own the accelerator after wrapper death.
        "lock=" + q(lock),
        "echo $$ > \"$lock/pid\"",
        "printf '%s\\n' " + q(boot) + ' > "$lock/boot_id"',
        "printf '%s\\n' " + q(identity) + ' > "$lock/identity"',
        "awk '{print $22}' /proc/$$/stat > \"$lock/start_ticks\"",
        'release_session_lock() { if test "$(cat "$lock/pid" 2>/dev/null)" = "$$"; then '
        'rm -f "$lock/pid" "$lock/boot_id" "$lock/identity" "$lock/start_ticks"; rmdir "$lock"; fi; }',
        # Some sh implementations run EXIT on a fatal signal while a foreground
        # child survives. Never release ownership on that path. A trap may be
        # deferred until the child finishes; both cases retain the boot's lock.
        'abandon_session_lock() { trap - EXIT; exit "$1"; }',
        "trap 'abandon_session_lock 129' HUP",
        "trap 'abandon_session_lock 130' INT",
        "trap 'abandon_session_lock 131' QUIT",
        "trap 'abandon_session_lock 143' TERM",
        "trap release_session_lock EXIT",
    ]


def dma_check(expected_bytes):
    return [
        'test -c /dev/udmabuf0 || { echo "missing /dev/udmabuf0" >&2; exit 1; }',
        'test "$(cat /sys/class/u-dma-buf/udmabuf0/size)" = ' + str(expected_bytes) +
        ' || { echo "DMA buffer size mismatch" >&2; exit 1; }',
        'test -r /sys/class/u-dma-buf/udmabuf0/phys_addr',
    ]


def session_script(base, remote, boot, identity, argv):
    q = shlex.quote
    return "\n".join([
        "#!/bin/sh", "cd " + q(remote) + " || exit 97", "mkdir .running 2>/dev/null || exit 98",
        "echo $$ > .running/pid", "printf '%s\\n' " + q(boot) + " > boot_id",
        "printf '%s\\n' " + q(identity) + " > task_hash",
        *session_lock(base, boot, identity, "echo 95 > exit_code; exit 95"),
        "renice -n 0 -p $$ > nice.log 2>&1 || { echo 94 > exit_code; exit 94; }",
        "printf '0' > /sys/class/fpga_manager/fpga0/flags || { echo 96 > exit_code; exit 96; }",
        "printf 'vta_hpc.bit\\n' > /sys/class/fpga_manager/fpga0/firmware || { echo 96 > exit_code; exit 96; }",
        'test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating || { echo 96 > exit_code; exit 96; }',
        shlex.join(argv) + " > native.log 2>&1", "code=$?",
        # Release ownership before publishing completion so acknowledgement cannot
        # delete files underneath a still-running wrapper.
        "release_session_lock", "trap - EXIT",
        "echo $code > exit_code.tmp", "mv exit_code.tmp exit_code",
    ]) + "\n"
