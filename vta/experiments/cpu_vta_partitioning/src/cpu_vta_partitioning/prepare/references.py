"""Require independent ARM references for the ARM floating-point stem."""
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import digest, read, sha256, write


TARGET = "AArch64 CPU LLVM; canonical full graph and independent complete frontier probe"
PROOF_STATUS = "ARM_COMPLETE_GRAPH_PROBE_BYTE_EXACT_67"


def _sealed(document):
    if document.get("artifact_sha256") != digest({k: v for k, v in document.items()
                                                  if k != "artifact_sha256"}):
        raise ValueError("reference provenance seal mismatch")


def _documents(manifest, proof, prior, canonical_hash, manifest_hash):
    for document in (manifest, proof, prior):
        _sealed(document)
    if manifest.get("reference_execution_target") != TARGET:
        raise ValueError("independent AArch64 references required; x86 float stem is not bit-exact on ARM")
    if (manifest.get("canonical_qir_sha256") != canonical_hash or
            prior.get("canonical_qir_sha256") != canonical_hash):
        raise ValueError("reference canonical graph mismatch")
    if (proof.get("status") != PROOF_STATUS or proof.get("new_performance_windows") != 0 or
            proof.get("inputs_manifest", {}).get("sha256") != manifest_hash or
            set(proof.get("sessions", {})) != {"complete_graph", "complete_frontier_probe"}):
        raise ValueError("missing independent complete graph/frontier reference proof")
    if manifest.get("inputs_changed") is not False or manifest.get("extra_quantize_calls") != 0:
        raise ValueError("ARM reference must preserve inputs and quantization")
    for group, count in (("real_inputs", 64), ("diagnostic_inputs", 3)):
        current, original = manifest[group], prior[group]
        if len(current) != count or len(original) != count:
            raise ValueError("incomplete reference inputs")
        if any((a["id"], a["input"]["sha256"]) != (b["id"], b["input"]["sha256"])
               for a, b in zip(current, original)):
            raise ValueError("reference input identity/order changed")


def freeze_reference_source(source, workspace, canonical_hash):
    """Check original provenance locally, then freeze documents without path dependencies."""
    source, workspace = Path(source), Path(workspace)
    manifest = read(source)
    if manifest.get("reference_execution_target") != TARGET:
        raise ValueError("independent AArch64 references required; x86 float stem is not bit-exact on ARM")
    proof_path = source.with_name("reference_proof.json")
    prior_ref = manifest["prior_x86_reference_manifest"]
    prior_path = Path(prior_ref["path"])
    if sha256(prior_path) != prior_ref["sha256"]:
        raise ValueError("prior input manifest hash mismatch")
    _documents(manifest, read(proof_path), read(prior_path), canonical_hash, sha256(source))
    artifacts = manifest["reference_artifacts"]
    if not artifacts:
        raise ValueError("missing independent reference program provenance")
    for record in artifacts:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("reference artifact hash mismatch: " + record["path"])
    destination = workspace / "sources/reference"
    destination.mkdir(parents=True, exist_ok=True)
    documents = {}
    for name, path in (("inputs_manifest.json", source), ("reference_proof.json", proof_path),
                       ("prior_inputs_manifest.json", prior_path)):
        target = destination / name
        expected = sha256(path)
        if target.exists() and sha256(target) != expected:
            raise ValueError("frozen reference provenance changed")
        if not target.exists():
            shutil.copy2(path, target)
        if sha256(target) != expected:
            raise ValueError("reference source changed during copy")
        documents[str(target.relative_to(workspace))] = expected
    identity = {"target": TARGET, "canonical_sha256": canonical_hash,
                "proof_status": PROOF_STATUS, "documents": documents,
                "reference_program_artifacts": artifacts,
                "historical_performance_used": False, "inputs_changed": False}
    write(workspace / "plans/reference_identity.json", identity, replace=True)
    return manifest, identity


def validate_frozen_reference(bundle):
    """Validate a relocated bundle using only its own documents."""
    bundle = Path(bundle)
    identity = read(bundle / "plans/reference_identity.json")
    expected = {"sources/reference/" + name for name in
                ("inputs_manifest.json", "reference_proof.json", "prior_inputs_manifest.json")}
    if set(identity["documents"]) != expected or identity["target"] != TARGET:
        raise ValueError("invalid frozen reference provenance")
    for name, value in identity["documents"].items():
        if sha256(bundle / name) != value:
            raise ValueError("frozen reference document hash mismatch")
    source = bundle / "sources/reference/inputs_manifest.json"
    canonical_hash = sha256(bundle / "sources/canonical_qir.json")
    if identity["canonical_sha256"] != canonical_hash:
        raise ValueError("frozen reference canonical graph mismatch")
    manifest = read(source)
    prior_path = bundle / "sources/reference/prior_inputs_manifest.json"
    if sha256(prior_path) != manifest["prior_x86_reference_manifest"]["sha256"]:
        raise ValueError("frozen prior input manifest hash mismatch")
    _documents(manifest, read(source.with_name("reference_proof.json")), read(prior_path),
               canonical_hash, sha256(source))
    return identity
