#!/usr/bin/env python3
"""Plot VTA fsim scaling statistics collected from deploy_classification_optimized.py."""

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Collected from fsim runs of resnet18_v1 on VTA.
    input_size = np.array([224, 256, 320, 384, 448, 512, 576], dtype=np.int32)

    inp_load_nbytes = np.array(
        [5549568, 116175360, 128952832, 164502016, 204376576, 248576512, 297101824],
        dtype=np.int64,
    )
    wgt_load_nbytes = np.array(
        [12763136, 2011430912, 2214592512, 2802843648, 3460300800, 4186963968, 4982833152],
        dtype=np.int64,
    )
    acc_load_nbytes = np.array(
        [6051840, 14773248, 16646144, 21067776, 26009600, 31471616, 37453824],
        dtype=np.int64,
    )
    out_store_nbytes = np.array(
        [2433536, 2821376, 3178496, 4022784, 4966400, 6009344, 7151616],
        dtype=np.int64,
    )
    gemm_counter = np.array(
        [6623232, 7857152, 8650752, 10948608, 13516800, 16355328, 19464192],
        dtype=np.int64,
    )
    alu_counter = np.array(
        [699328, 810240, 913408, 1156032, 1427200, 1726912, 2055168],
        dtype=np.int64,
    )

    total_moved_bytes = inp_load_nbytes + wgt_load_nbytes + acc_load_nbytes + out_store_nbytes

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.30, wspace=0.26)

    ax = axes[0, 0]
    ax.plot(input_size, inp_load_nbytes / 1e6, marker="o", linewidth=2, label="inp_load")
    ax.plot(input_size, wgt_load_nbytes / 1e6, marker="s", linewidth=2, label="wgt_load")
    ax.plot(input_size, acc_load_nbytes / 1e6, marker="^", linewidth=2, label="acc_load")
    ax.plot(input_size, out_store_nbytes / 1e6, marker="d", linewidth=2, label="out_store")
    ax.set_title("(a) Data Movement vs Input Size")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Bytes per Run (MB)")
    ax.legend(frameon=True)

    ax = axes[0, 1]
    ax.plot(input_size, gemm_counter / 1e6, marker="o", linewidth=2, label="gemm_counter")
    ax.plot(input_size, alu_counter / 1e6, marker="s", linewidth=2, label="alu_counter")
    ax.set_title("(b) Compute Counters vs Input Size")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Counter Value (Million)")
    ax.legend(frameon=True)

    ax = axes[1, 0]
    ax.plot(
        input_size[1:],
        (wgt_load_nbytes[1:] / gemm_counter[1:]),
        marker="o",
        linewidth=2,
        label="wgt_load / gemm",
    )
    ax.plot(
        input_size[1:],
        (total_moved_bytes[1:] / gemm_counter[1:]),
        marker="s",
        linewidth=2,
        label="total_bytes / gemm",
    )
    ax.set_title("(c) Bytes per GEMM Counter")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Bytes / GEMM Count")
    ax.legend(frameon=True)

    ax = axes[1, 1]
    base = total_moved_bytes[1]
    ax.plot(
        input_size[1:],
        total_moved_bytes[1:] / base,
        marker="o",
        linewidth=2,
        label="total movement growth",
    )
    ax.plot(
        input_size[1:],
        gemm_counter[1:] / gemm_counter[1],
        marker="s",
        linewidth=2,
        label="gemm growth",
    )
    ax.plot(
        input_size[1:],
        alu_counter[1:] / alu_counter[1],
        marker="^",
        linewidth=2,
        label="alu growth",
    )
    ax.set_title("(d) Growth Relative to 256")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Normalized Ratio")
    ax.legend(frameon=True)

    for ax in axes.flat:
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig("vta_fsim_scaling_stats.png", dpi=300, bbox_inches="tight")
    plt.savefig("vta_fsim_scaling_stats.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
