# Unselected UB permutation prototype

This alternative loads all selected experts into UB, permutes using Gather, and writes contiguous output. Both serial and overlapped variants are exact, but full Gather+up was around 60.7 us, slower than the selected direct-copy implementation at 52.6 us. It is not enabled in inference.

Source and the early 8-call graph control are retained. The 8-call standalone gather result reaches the host-launch floor and must not be presented as exact kernel duration. Use the 16-call/full-path comparison in benchmarks/r138-expert-gather/probe.json and HTTP acceptance.

The selected implementation originated in the earlier r128 experiment. Rechecking it with amortized graph launch controls was necessary: the historical approximately .30 ms single-replay MoE comparison was too close to host dispatch limits to reject its device-side benefit.
