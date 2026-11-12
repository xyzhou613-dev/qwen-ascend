import os
import delta_probe
f=delta_probe.E.delta_chunk_batched_fp32
def baseline(*args):
 os.environ['FNX_DELTA_REUSE']='0'
 return f(*args)
def candidate(*args):
 os.environ['FNX_DELTA_REUSE']='1'
 return f(*args)
delta_probe.E.delta_chunk_batched_fp32=baseline
delta_probe.candidate=candidate
delta_probe.main()
