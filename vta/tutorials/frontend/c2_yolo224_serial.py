#!/usr/bin/env python3
"""Recover only the verified board's transient IPv4 address through its UART.

Never resets the board, changes DTR/RTS, writes SD startup files, closes another
application, or attempts a password. A busy/unknown console requires the user.
"""
import argparse,ipaddress,json,re,subprocess
from pathlib import Path
from c2_yolo224_prepare import write,sha

POWERSHELL='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def exchange(out,port,text,label):
    if not re.fullmatch(r'COM[1-9][0-9]*',port):raise ValueError('invalid Windows serial port')
    # UART bytes are data, never interpolated as PowerShell source code.
    import base64
    payload=base64.b64encode(text.encode('ascii')).decode('ascii')
    script='''$ErrorActionPreference="Stop"
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$p=New-Object System.IO.Ports.SerialPort "PORT",115200,"None",8,"One"
$p.ReadTimeout=500; $p.WriteTimeout=1000; $p.DtrEnable=$false; $p.RtsEnable=$false
try {
  $p.Open(); Start-Sleep -Milliseconds 200
  Write-Output $p.ReadExisting()
  $bytes=[Convert]::FromBase64String("PAYLOAD")
  $p.Write($bytes,0,$bytes.Length)
  Start-Sleep -Milliseconds 1500
  Write-Output $p.ReadExisting()
} finally { if($p.IsOpen){$p.Close()} }
'''.replace('PORT',port).replace('PAYLOAD',payload)
    target=out/(label+'.json');index=1
    while target.exists():index+=1;target=out/(label+f'_{index:02d}.json')
    result=subprocess.run([POWERSHELL,'-NoProfile','-Command',script],capture_output=True,timeout=15)
    target.with_suffix('.stdout').write_bytes(result.stdout);target.with_suffix('.stderr').write_bytes(result.stderr)
    output=result.stdout.decode('utf-8',errors='replace')
    write(target,{'port':port,'baud':115200,'payload_ascii':text,'returncode':result.returncode,
        'stdout':output,'stderr':result.stderr.decode('utf-8',errors='replace'),
        'DTR':False,'RTS':False,'generator_sha256':sha(Path(__file__))})
    if result.returncode:raise RuntimeError('serial port unavailable; inspect '+str(target))
    return re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',output).replace('\r','')


def recover(out,port,ip,mac,cid):
    ip=str(ipaddress.IPv4Address(ip));mac=mac.lower();cid=cid.lower()
    if not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}',mac) or not re.fullmatch(r'[0-9a-f]{32}',cid):
        raise ValueError('missing pinned board identity')
    probe=exchange(out,port,'\r','serial_prompt')
    if re.search(r'(?im)login:\s*$',probe):probe=exchange(out,port,'root\r','serial_root_login')
    if re.search(r'(?im)password:\s*$',probe) or not re.search(r'(?m)[#$]\s*$',probe):
        raise RuntimeError('a Linux root shell is unavailable; user console intervention is required')
    command='printf "\\nC2_SERIAL_ID_BEGIN\\n"; cat /proc/sys/kernel/random/boot_id; '+\
        'cat /sys/class/net/eth0/address; cat /sys/block/mmcblk1/device/cid; printf "C2_SERIAL_ID_END\\n"\r'
    observed=exchange(out,port,command,'serial_identity')
    match=re.search(r'(?ms)^C2_SERIAL_ID_BEGIN\n(.*?)^C2_SERIAL_ID_END\s*$',observed)
    if not match:raise ValueError('actual board identity was not captured from the UART')
    values=[x.strip().lower() for x in match[1].splitlines() if x.strip()]
    if len(values)!=3 or values[1:]!=[mac,cid] or not re.fullmatch(r'[0-9a-f-]{36}',values[0]):
        raise ValueError('UART belongs to an unverified board; no network configuration sent')
    command='ifconfig eth0 '+ip+' netmask 255.255.255.0 up && printf "C2_TRANSIENT_NETWORK_RESTORED\\n"; ifconfig eth0\r'
    configured=exchange(out,port,command,'serial_restore_transient_IP')
    if not re.search(r'(?m)^C2_TRANSIENT_NETWORK_RESTORED\s*$',configured):raise ValueError('network restoration did not succeed')
    marker=out/'serial_network_recovery.json'
    value={'status':'passed','port':port,'baud':115200,'boot_id':values[0],'MAC':mac,'SD_CID':cid,
        'transient_IP':ip,'persistent_configuration_modified':False,'reset_or_password_attempted':False}
    if not marker.exists():write(marker,value)
    elif json.loads(marker.read_text())!=value:raise ValueError('serial recovery identity changed')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',default='COM3');p.add_argument('--ip',default='192.168.1.247')
    p.add_argument('--master',type=Path)
    p.add_argument('--mac',required=True);p.add_argument('--cid',required=True)
    a=p.parse_args();value=recover(a.out,a.port,a.ip,a.mac,a.cid)
    if a.master:
        path=a.master/'serial_connection.json'
        connection={'port':a.port,'baud':115200,'MAC':value['MAC'],'SD_CID':value['SD_CID'],
            'actual_verified_recovery':str(a.out/'serial_network_recovery.json')}
        if not path.exists():write(path,connection)
        elif json.loads(path.read_text())!=connection:raise ValueError('verified serial connection identity differs')
    print(json.dumps(value))
