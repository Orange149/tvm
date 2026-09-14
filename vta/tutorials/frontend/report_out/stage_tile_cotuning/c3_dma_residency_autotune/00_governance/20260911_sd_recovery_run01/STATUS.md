# SD-card recovery status

Status: **ext4 repaired and write-qualified; FAT cleanly remounted; cold-boot test pending**.

The board root is a RAM `rootfs`, so `/dev/mmcblk1p1` and `/dev/mmcblk1p2` were safely taken offline without stopping the active RAM-only VTA RPC service.

## Recovery point

Raw images were captured on the host before modifying the card:

| Image | Exact bytes | SHA-256 |
|---|---:|---|
| `/tmp/vta_sd_backup_20260911/mmcblk1.header-4MiB.bin` | 4,194,304 | `df812575399e6494b7b18c46a5e2fa55da5d840903655d7768249889be122297` |
| `/tmp/vta_sd_backup_20260911/mmcblk1p1.img` | 1,123,024,896 | `c7c2a5a61aeb3b4cbad6f4977f8c7dfdcb8e3ceb4d5c018c55439e4b21250161` |
| `/tmp/vta_sd_backup_20260911/mmcblk1p2.img` | 2,202,009,600 | `51ad9d8bbfb7bd0638b106df0037fcf028f450dd8077930bf2149a285095dd77` |

The earlier `.img.gz.partial` is an intentionally aborted, incomplete compression stream and is not a recovery image. The valid images are host-local `/tmp` artifacts and will not survive arbitrary host cleanup/reboot unless moved elsewhere.

## Root cause and repair

Read-only `e2fsck` on the untouched p2 image found 48 stale directory entries in `queue_q1_validation_run1/D_stress_outputs/run_923` through `run_970`: the entries referenced deleted/unused inodes 46945–46992, with matching block/inode bitmap and free-count differences. This is consistent with an interrupted, full-filesystem stress-output update; no block-device I/O error was observed during the complete raw backup.

The exact repair was first simulated on a copy of the image. After that test passed, ARM64 `e2fsck 1.45.3` from the matching PetaLinux SDK was executed from tmpfs against the unmounted `/dev/mmcblk1p2`. It cleared those 48 stale entries and repaired the bitmaps/free counts. An immediate second `e2fsck -fn` returned 0 with all five passes clean.

Two regenerable, inactive cache directories were then removed to solve the full-volume condition:

- `/media/sd-mmcblk1p2/_file_cache` — 208,756 KiB;
- `/media/sd-mmcblk1p2/v1_p7_top20/_file_cache` — 284,856 KiB.

The current RPC working directory was verified as `/var/volatile/vta_c3_ram/runtime`, so neither cache was active. Both directories remain recoverable from the raw p2 image. P2 usage fell from 100%/0 free to 78%/381.2 MiB free.

## Qualification

- P2 64 MiB zero-file write, sync, exact size and SHA-256 check: passed; temporary file deleted.
- P2 clean unmount followed by full `e2fsck -fn`: return code 0.
- P1 1 MiB write, sync, exact size and SHA-256 check: passed; temporary file deleted.
- P1 clean unmount and read-only remount: no new FAT, MMC, buffer-I/O or I/O error message.
- P1 did not receive a structural `fsck.fat`, because neither board rootfs nor the local SDK contains that utility. Its observed issue was an unclean-unmount warning, and the clean write/unmount/remount cycle completed without a repeated warning.

The card is qualified for continued experiments on the current boot. Experiment executables and RPC uploads should still use tmpfs to avoid unnecessary SD wear and storage-dependent timing. A later user-controlled reboot is required to certify the cold-boot path; it is not required for the current RAM-resident P7 experiment.
