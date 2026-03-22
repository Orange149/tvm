#!/usr/bin/env python3
"""Plot VTA input-scaling profiling results for thesis figures."""

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Collected from AXU5EVB runs with deploy_classification_optimized.py
    input_size = np.array([224, 256, 320, 384])

    set_data_ms = np.array([60.833, 79.564, 133.974, 185.402])
    run_ms = np.array([111.691, 1090.723, 1618.868, 2265.928])
    device_run_wait_ms = np.array([81.424, 964.255, 1459.890, 2047.645])
    mem_copy_from_host_ms = np.array([1.823, 2.385, 3.712, 5.357])

    load_mb = np.array([17.49, 170.53, 229.10, 294.05])
    store_mb = np.array([1.60, 2.09, 3.27, 4.71])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.28)

    # (a) End-to-end latency breakdown
    ax = axes[0, 0]
    ax.plot(input_size, set_data_ms, marker="o", linewidth=2, label="set_data")
    ax.plot(input_size, run_ms, marker="s", linewidth=2, label="run")
    ax.plot(input_size, device_run_wait_ms, marker="^", linewidth=2, label="device_run_wait")
    ax.set_title("(a) Latency vs Input Resolution")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Time (ms)")
    ax.legend(frameon=True)

    # (b) Host-side transfer cost
    ax = axes[0, 1]
    ax.plot(input_size, mem_copy_from_host_ms, marker="o", linewidth=2, label="mem_copy_from_host")
    ax.plot(input_size, set_data_ms, marker="s", linewidth=2, label="set_data")
    ax.set_title("(b) Host-Side Input Transfer")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Time (ms)")
    ax.legend(frameon=True)

    # (c) Internal VTA data movement
    ax = axes[1, 0]
    ax.plot(input_size, load_mb, marker="o", linewidth=2, label="load_buffer_2d")
    ax.plot(input_size, store_mb, marker="s", linewidth=2, label="store_buffer_2d")
    ax.set_title("(c) VTA Internal Data Movement per Run")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Transferred Data (MB)")
    ax.legend(frameon=True)

    # (d) Device execution wait vs graph run
    ax = axes[1, 1]
    bar_width = 12
    ax.bar(input_size - bar_width / 2, run_ms, width=bar_width, label="run")
    ax.bar(input_size + bar_width / 2, device_run_wait_ms, width=bar_width, label="device_run_wait")
    ax.set_title("(d) Runtime vs Device Execution Wait")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Time (ms)")
    ax.legend(frameon=True)

    for ax in axes.flat:
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig("vta_input_scaling_profile.png", dpi=300, bbox_inches="tight")
    plt.savefig("vta_input_scaling_profile.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
