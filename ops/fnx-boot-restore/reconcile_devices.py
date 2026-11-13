"""Reconcile only the original Flash container's two known PCI devices.

Caller must stop dockerd and rule out other running containers first.
Raw HostConfig backup stays private. Existing container/layers are preserved.
"""
import argparse,hashlib,json,os,re,stat,subprocess,time
from pathlib import Path
def restore_intended(running,error,manually_stopped):
 return not manually_stopped and (running or ('device' in error.lower() and '/dev/davinci' in error))
def selected_devices(text):
 pairs=re.findall(r'\|\s+(\d+)\s+910B4-1\s+\|[^\n]*\n\|\s+\d+\s+\|\s+([0-9a-fA-F:.]+)',text)
 chosen={bus.lower():int(number) for number,bus in pairs if bus.lower() in ('0000:83:00.0','0000:03:00.0')}
 assert set(chosen)=={'0000:83:00.0','0000:03:00.0'},('EXPECTED_PCI_PAIR_NOT_FOUND',chosen)
 return sorted(chosen.values()),chosen
def patch_devices(config,numbers):
 result=json.loads(json.dumps(config));items=result['Devices']
 slots=[r for r in items if re.fullmatch('/dev/davinci[0-9]+',r['PathOnHost'])]
 assert len(slots)==2 and all(r['PathInContainer']==r['PathOnHost'] for r in slots)
 for row,n in zip(slots,numbers):row['PathOnHost']=row['PathInContainer']=f'/dev/davinci{n}'
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--state',required=True);ap.add_argument('--apply',action='store_true');a=ap.parse_args();p=Path(a.state)
 identity=json.load(open(p/'container-identity.json'));root=Path(identity['docker_root']).resolve();cid=identity['container_id']
 assert re.fullmatch('[0-9a-f]{64}',cid) and root.is_absolute()
 target=root/'containers'/cid/'hostconfig.json';old=json.load(open(target))
 text=subprocess.check_output(['npu-smi','info'],text=True);numbers,buses=selected_devices(text)
 new=patch_devices(old,numbers)
 for row in new['Devices']:
  assert stat.S_ISCHR(os.stat(row['PathOnHost']).st_mode),('MISSING_DEVICE',row['PathOnHost'])
 result=dict(timestamp_epoch=time.time(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pci_indices=buses,
  before=old['Devices'],after=new['Devices'],changed=old['Devices']!=new['Devices'],container_recreated=False)
 (p/'device-plan.json').write_text(json.dumps(result,indent=2))
 if a.apply and result['changed']:
  state=subprocess.check_output(['systemctl','show','docker.service','--value','-p','ActiveState'],text=True).strip();assert state=='inactive',state
  private=p/'private';private.mkdir(exist_ok=True);private.chmod(0o700)
  backup=private/('hostconfig-'+str(time.time_ns())+'.json');backup.write_bytes(target.read_bytes());backup.chmod(0o600)
  metadata=target.stat();temp=target.with_suffix('.fnx-device.tmp')
  with temp.open('w') as f:json.dump(new,f,separators=(',',':'));f.flush();os.fsync(f.fileno())
  os.chmod(temp,stat.S_IMODE(metadata.st_mode));os.chown(temp,metadata.st_uid,metadata.st_gid);os.replace(temp,target)
 print('DEVICE_RECONCILE',json.dumps(result))
if __name__=='__main__':main()
