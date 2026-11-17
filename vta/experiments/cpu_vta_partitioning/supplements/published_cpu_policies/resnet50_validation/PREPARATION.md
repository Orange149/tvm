# Local ResNet50 preparation

The preparation modules never invoke SSH/SCP. They build the new network's
qualified intervals while retaining the existing runtime, executor, bitstream
and driver identities. Run the public entry point from the parent package; the
Python interfaces below are also available to its workflow.

- `assets.audit_assets(local_config)` verifies the pretrained Gluon
  `resnet50_v2` weights, the input-zero tensor and the six reused runtime assets.
  It imports neither TVM nor a board transport.
- `prepare.build(output, local_config, names=None)` builds the declared program
  list with at most four SDK workers. A new compiler starts only with at least
  4 GiB host memory available; owned compiler process groups are stopped if
  available host memory drops below the 2 GiB reserve. The list contains 613 search intervals plus
  the two independent CPU reference programs. A record is reused only after its
  source identity and all compiled bytes match. Failure logs are retained and a
  failed interval does not silently disappear from the domain.
- `prepare.prepare_references(output)` executes the complete quantized ARM graph
  and the complete-frontier probe locally with QEMU. Their terminal output must
  be byte identical. All frontier tensors are saved for subsequent isolated
  stage measurements. The proof records QEMU, not board execution or accuracy.
- `prepare.prepare_profile(output)` seals a relative-path profile capsule only
  after all declared modules and the independent ARM references validate.
  Its first 39 component signatures are the six structural CPU representatives
  at four thread counts and three VTA representatives at five host placements.
  `plans/pilot.json` identifies the same three VTA modules for DMA accounting.
- `prepare.validate_build` and `prepare.validate_profile` reject missing,
  changed, ambiguous or incorrectly sourced modules instead of recompiling or
  substituting same-named files during validation.

## Schedule and source scope

The compiler preserves the existing ResNet18/YOLO `ApplyHistoryBest` policy:
match the complete template/workload by model, then by target key. Every lookup
records its requested target, chosen source record, configuration and selection
path in `schedule_history.json`; unmatched requests use the framework default
and are explicitly marked. The native dispatcher source is frozen and checked.
There is no new tuning or import of kernel-history timings as stage costs.
The compiler closure, common QIR, qconfig, input, SDK and host-library identity
are frozen before building.

The initial `resnet50_v2_validation_01` compilation incorrectly required whole
target-string equality. It rejected all 22 available VTA workload matches and
is retained as a stopped regression batch. Recovery uses the separate
`resnet50_v2_validation_native_history_01` directory and verified copies of the
same graph/unit evidence. Old module records are never relabelled as newly
compiled modules. All modules are recompiled under the corrected identity;
the 17 CPU modules with no AutoTVM queries cost only 28.16 prior worker-seconds,
so their negligible rebuild avoids an unnecessary provenance exception.

The executable profile records contain only `graph.json`, `module.so` and
`params.bin`. Nonexecution lowering/history proofs stay in the immutable build
workspace, with their identities and original record hash retained in capsule
records. The pipeline bundle can therefore copy the compact records without
claiming absent proof files are runtime dependencies.

## Memory and QEMU qualification

Visible graph tensor storage is computed as the maximum tensor bytes for each
`(device_index, storage_id)` pair. Host parameter blobs and copied boundary
queues are reported separately; they are not added to ext-dev demand. The
profile compares only visible device-type-12 tensor slots with the existing
192 MiB device buffer. Runtime/TIR workspace, allocator effects and actual
residency remain unknown, so this is neither a measured peak nor a strict bound.
The first actual board qualification still has to establish successful loading
and allocation.

QEMU creates an auxiliary host thread outside the logical TVM worker pool.
Local reference qualification uses the existing common-comparison rule:
all active logical slots and caller affinities must match exactly; an auxiliary
is ignored only when its recorded CPU-time increment is exactly zero. This rule
does not relax board validators. The initial generic-affinity preflight failure
is preserved separately; local reference attempts and accepted calls are both
accounted for in `LOCAL_REFERENCE_AUDIT.json`.
