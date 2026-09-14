import json
import pytest
from vta_geometry_catalog import audit_sources, assert_unreserved


def test_later_nested_contract_blocks_legacy_blind_spot(tmp_path):
    legacy=tmp_path/'legacy.json';legacy.write_text('{}')
    later=tmp_path/'later';later.mkdir()
    sig=[1,256,14,14,1024,1,1,1,1,0,0,0,0]
    (later/'board_collection_contract.json').write_text(json.dumps({'nested':{'geometry_signature':sig}}))
    rows=audit_sources(tmp_path,[legacy])
    with pytest.raises(ValueError,match='already reserved'):
        assert_unreserved(sig,rows)
    assert_unreserved([1,512,14,14,1024,1,1,1,1,0,0,0,0],rows)


def test_failed_contract_workload_is_still_reserved(tmp_path):
    w=['conv2d_packed.vta',['TENSOR',[1,16,14,14,1,16],'int8'],
       ['TENSOR',[64,16,1,1,16,16],'int8'],[1,1],[0,0,0,0]]
    path=tmp_path/'failed_contract.json';path.write_text(json.dumps({'status':'failed','items':[{'workload':w}]}))
    rows=audit_sources(tmp_path,[])
    with pytest.raises(ValueError):assert_unreserved([1,256,14,14,1024,1,1,1,1,0,0,0,0],rows)


def test_unreadable_or_malformed_source_does_not_silently_pass(tmp_path):
    (tmp_path/'bad_contract.json').write_text('{')
    with pytest.raises(json.JSONDecodeError):audit_sources(tmp_path,[])
    with pytest.raises(FileNotFoundError):audit_sources(tmp_path/'other',[tmp_path/'missing.json'])
