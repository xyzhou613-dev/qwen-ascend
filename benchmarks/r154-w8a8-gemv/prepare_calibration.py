"""Freeze a development NLL suite; not the final quality acceptance holdout."""
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent;root=P/'suite-sources'
tok=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
old=json.load(open(root/'fixtures.json'))
math_text='''To count the subsets of a finite set with n elements, decide independently whether each element belongs to the subset. There are two choices at each of n positions, so there are 2^n subsets in total. If only subsets of size k are allowed, count ordered choices of k distinct elements first. There are n(n-1)...(n-k+1) such ordered lists. Every subset of size k occurs in exactly k! orders. Dividing by k! gives the binomial coefficient C(n,k)=n!/(k!(n-k)!). Partitioning all subsets according to their sizes proves that the sum of C(n,k), for k from zero through n, equals 2^n. The empty subset accounts for the k=0 term, and the full set accounts for the k=n term.
Another proof uses the binomial theorem. Expand (1+x)^n by choosing either 1 or x from each factor. A term x^k arises by choosing x from exactly k factors, so its coefficient is C(n,k). Substituting x=1 gives the same identity. Substituting x=-1 instead shows that, when n is positive, the sum of the even-indexed coefficients equals the sum of the odd-indexed coefficients. A direct bijection explains this equality: choose one distinguished element and toggle its membership in a subset. The operation changes the parity of the subset size and is its own inverse.
For a finite geometric series, write S=1+r+r^2+...+r^m. Multiplying by r and subtracting gives (1-r)S=1-r^(m+1). If r is not one, division yields S=(1-r^(m+1))/(1-r). If r=1, every term equals one, so S=m+1. For absolute value of r less than one, r^(m+1) approaches zero as m grows, and the infinite series converges to 1/(1-r). The finite identity is exact and does not require a convergence argument.'''
spec=[('cal_natural_a','natural18000',512),('cal_natural_b','natural36000',4096)]
cases=[]
def add(name,ids,source,source_hash=None):
 ids=ids[:129];assert len(ids)==129,(name,len(ids))
 cases.append(dict(name=name,ids=ids,source=source,source_sha256=source_hash,
     ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),warm_context=0,scored_tail=128))
for name,key,offset in spec:add(name,old[key][offset:offset+129],f'r152 frozen {key}, offset {offset}')
files=[
       ('cal_python',root/'model.py',2000),
       ('cal_cpp',root/'gather.cpp',512),

       ]
for name,file,offset in files:
 text=file.read_text(encoding='utf-8-sig');ids=tok.encode(text)
 add(name,ids[offset:offset+129],str(file),hashlib.sha256(file.read_bytes()).hexdigest())

(P/'calibration-inputs.json').write_text(json.dumps(dict(cases=cases,scope='development suite, not final holdout',scored_targets=512),indent=2))
print('CALIBRATION_FROZEN',len(cases),512,flush=True)
