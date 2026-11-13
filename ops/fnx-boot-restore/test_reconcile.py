from reconcile_devices import patch_devices,restore_intended,selected_devices
base={'Devices':[dict(PathOnHost='/dev/davinci1',PathInContainer='/dev/davinci1',CgroupPermissions='rwm'),dict(PathOnHost='/dev/davinci0',PathInContainer='/dev/davinci0',CgroupPermissions='rwm'),dict(PathOnHost='/dev/davinci_manager',PathInContainer='/dev/davinci_manager',CgroupPermissions='rwm')],'Other':['preserved']}
new=patch_devices(base,[3,5]);assert base['Devices'][0]['PathOnHost']=='/dev/davinci1'
assert [x['PathOnHost'] for x in new['Devices']]==['/dev/davinci3','/dev/davinci5','/dev/davinci_manager'] and new['Other']==base['Other']
assert patch_devices(new,[0,1])['Devices'][1]['PathInContainer']=='/dev/davinci1'
assert restore_intended(True,'',False)
assert restore_intended(False,'error gathering device /dev/davinci3',False)
assert not restore_intended(False,'error gathering device /dev/davinci3',True)
assert not restore_intended(False,'',False)
assert not restore_intended(False,'unrelated error',False)
sample='| 3 910B4-1 | OK |\n| 0 | 0000:83:00.0 |\n| 5 910B4-1 | OK |\n| 0 | 0000:03:00.0 |'
assert selected_devices(sample)[0]==[3,5]
print('DEVICE_MAPPING_AND_RUNNING_INTENT_PASS')
