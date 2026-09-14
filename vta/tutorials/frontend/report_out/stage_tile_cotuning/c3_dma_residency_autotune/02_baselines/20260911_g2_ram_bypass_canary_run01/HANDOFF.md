# Handoff

The board can currently be used for narrowly scoped RAM-only correctness and pilot experiments under boot ID `a68a7983-719f-47bf-94d7-41f974c5342c`.

Keep `/dev/mmcblk1p2` read-only. Do not read experiment executables, libraries, bitstreams, inputs, outputs, or RPC uploads from that filesystem. Before each board batch, verify the boot ID, FPGA `operating` state, `/dev/udmabuf0` size, RPC PID, and that no new storage error has appeared after dmesg timestamp 467. A reboot destroys the RAM deployment and invalidates this current-boot qualification.

The next permitted step is the already frozen P6b 9-candidate mechanism correctness canary. Performance timing remains conditional on all three correctness seeds passing for a candidate.
