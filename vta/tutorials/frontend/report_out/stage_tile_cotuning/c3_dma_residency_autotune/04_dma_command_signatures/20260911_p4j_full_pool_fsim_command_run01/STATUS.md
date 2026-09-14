# C3-P4j full-pool local FSim command dry-run

Status: **protocol deviation; superseded and not final evidence**. This run used TVM's local-only `rpc.LocalSession()` API. It did not use a network, SSH, or a board, and its 197/197 correctness result remains an audit trail, but it violates P4j's strict no-RPC protocol. Only `20260911_p4j_full_pool_fsim_command_run02` may be consumed downstream.
