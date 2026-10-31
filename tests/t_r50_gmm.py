"""Test the actual per-group int4 GMM constraints before implementing routing."""
import torch
import torch_npu


def main():
    torch.npu.set_device(0)
    E,K,N,M=4,256,128,8
    x=torch.randn(M*E,K,dtype=torch.bfloat16,device='npu:0')
    q=torch.randint(-8,8,(E,K,N),dtype=torch.int32,device='npu:0')
    w=torch_npu.npu_convert_weight_to_int4pack(q.reshape(-1,N)).view(E,K,N//8)
    ends=torch.arange(1,E+1,device='npu:0',dtype=torch.int64)*M
    for rank in (2,3):
        scales=torch.ones((E,N) if rank==2 else (E,K//64,N),device='npu:0',dtype=torch.bfloat16)*.01
        for layout in ('stack','list'):
            try:
                weights=[w] if layout=='stack' else list(w.unbind(0))
                sc=[scales] if layout=='stack' else list(scales.unbind(0))
                out=torch_npu.npu_grouped_matmul([x],weights,antiquant_scale=sc,
                          group_list=ends,split_item=3,group_type=0,output_dtype=torch.bfloat16)
                torch.npu.synchronize()
                print('GMM_PASS',rank,layout,[tuple(y.shape) for y in out],flush=True)
            except Exception as ex:print('GMM_FAIL',rank,layout,str(ex)[:1000],flush=True)


if __name__=='__main__':
    with torch.no_grad(): main()
