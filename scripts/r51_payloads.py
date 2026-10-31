"""Prepare exact near-8k prompts; run measurements from the Windows client."""
import json,uuid,os
from pathlib import Path
from transformers import AutoTokenizer

tk=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
text=Path('/root/ascend-dev/eval_text.txt').read_text(encoding='utf-8')
filler=tk((text+'\n')*100).input_ids
out=Path(os.environ.get('BENCH_DIR','/root/ascend-dev/r51'));out.mkdir(exist_ok=True)
for i in range(3):
    prefix='本次性能测量唯一编号：'+uuid.uuid4().hex+'。\n参考资料：\n'
    suffix='\n参考资料结束。请结合资料写一篇不少于800字的技术说明，详细解释推理服务如何区分首token延迟、prefill吞吐和decode速度。直接连续写正文，不要只给结论。'
    n=7900
    for _ in range(12):
        content=prefix+tk.decode(filler[:n])+suffix
        messages=[{'role':'user','content':content}]
        template=tk.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tk(template).input_ids
        if len(ids)==8064:break
        n+=8064-len(ids)
    assert len(ids)==8064,len(ids)
    payload={'messages':messages,'stream':True,'max_tokens':128,'temperature':0,'enable_thinking':False}
    (out/f'payload{i+1}.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    print('PAYLOAD',i+1,'prompt_tokens',len(ids),'max_output',128,flush=True)
