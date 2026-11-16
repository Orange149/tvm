import ast
from pathlib import Path
from cpu_vta_partitioning.common import PROJECT


def test_offline_modules_have_no_transport_or_remote_commands():
    for directory in ('prepare','adapters','domain','models','search','analysis','measurement'):
        for path in (PROJECT/'src/cpu_vta_partitioning'/directory).rglob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    assert 'transport' not in (node.module or '')
                if isinstance(node,ast.Import):
                    assert all('transport' not in item.name for item in node.names)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('run','Popen','check_call','check_output'):
                    if node.args and isinstance(node.args[0],(ast.List,ast.Tuple)) and node.args[0].elts:
                        first=node.args[0].elts[0]
                        if isinstance(first,ast.Constant):
                            assert Path(str(first.value)).name not in ('ssh','scp','sftp')


def test_cli_offline_dispatch_does_not_instantiate_transport(monkeypatch,tmp_path):
    from cpu_vta_partitioning import cli
    from cpu_vta_partitioning import transport
    from cpu_vta_partitioning.prepare import package
    def forbidden(*args,**kwargs):
        raise AssertionError('offline entry attempted board connection')
    monkeypatch.setattr(transport,'SSH',forbidden)
    monkeypatch.setattr(cli,'local_prepare',lambda *args:tmp_path/'bundle')
    monkeypatch.setattr(package,'validate',lambda *args:{'status':'VALIDATED_OFFLINE'})
    assert cli.main(['prepare','--config','config.json','--local-config','local.json'])==0
    assert cli.main(['validate','--bundle',str(tmp_path/'bundle')])==0
