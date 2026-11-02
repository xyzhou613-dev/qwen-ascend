from pathlib import Path
import shutil
r=Path('/root/ascend-dev');s=r/'r94';s.mkdir(exist_ok=True)
shutil.copytree(r/'fnx/fnx',s/'fnx',dirs_exist_ok=True)
p=s/'fnx/engine.py';text=p.read_text()
needle='        group_ids = pos // 4\n        rows = (group_ids[:, None] * 4 + torch.arange(4, device=self.dev)).reshape(-1)'
assert text.count(needle)==1
text=text.replace(needle,'''        if os.environ.get("FNX_INDEX_UNIQUE", "0") == "1" and raw.shape[0] > 1:
            # Fixed upper bound on touched groups is capture-safe for every
            # start alignment. Clamp the optional extra row to the last group;
            # do not mutate a causally future cache row.
            count = (raw.shape[0] + 6) // 4
            group_ids = (pos[0] // 4 + torch.arange(count, device=self.dev)).clamp_max(pos[-1] // 4)
        else:
            group_ids = pos // 4
        rows = (group_ids[:, None] * 4 + torch.arange(4, device=self.dev)).reshape(-1)''')
needle='''            tokens = (chosen[:, :, None] * 4 + torch.arange(4, device=self.dev)).reshape(end - begin, -1)
            valid = tokens < ((p[:, None] + 1) // 4) * 4
            tail = ((p[:, None] + 1) // 4) * 4 + torch.arange(3, device=self.dev)
            tokens = torch.cat((tokens, tail), -1)
            valid = torch.cat((valid, tail <= p[:, None]), -1)
            # Physical order reproduces dense attention's accumulation order.
            ordered = torch.where(valid, tokens, P).float().sort(-1).values.to(torch.int64)
'''
assert text.count(needle)==1
text=text.replace(needle,'''            if os.environ.get("FNX_INDEX_BLOCK_SORT", "0") == "1":
                # Sort groups before expanding their four adjacent tokens.
                complete_groups = chosen * 4 + 4 <= p[:, None] + 1
                groups_sorted = torch.where(complete_groups, chosen, P // 4).float().sort(-1).values.to(torch.int64)
                valid_width = complete_groups.sum(-1, keepdim=True) * 4
                columns = torch.arange(chosen.shape[1] * 4 + 3, device=self.dev)[None, :].expand(end-begin, -1)
                grouped = groups_sorted.gather(1, (columns // 4).clamp_max(chosen.shape[1]-1)) * 4 + columns % 4
                offset = columns - valid_width
                tail = ((p[:, None] + 1) // 4) * 4 + offset
                valid_tail = (offset >= 0) & (offset < 3) & (tail <= p[:, None])
                ordered = torch.where(columns < valid_width, grouped, torch.where(valid_tail, tail, P))
            else:
                tokens = (chosen[:, :, None] * 4 + torch.arange(4, device=self.dev)).reshape(end - begin, -1)
                valid = tokens < ((p[:, None] + 1) // 4) * 4
                tail = ((p[:, None] + 1) // 4) * 4 + torch.arange(3, device=self.dev)
                tokens = torch.cat((tokens, tail), -1)
                valid = torch.cat((valid, tail <= p[:, None]), -1)
                # Physical order reproduces dense attention's accumulation order.
                ordered = torch.where(valid, tokens, P).float().sort(-1).values.to(torch.int64)
''')
p.write_text(text);compile(text,str(p),'exec')
print('R94_PREPARED',flush=True)
