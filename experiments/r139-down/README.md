# Additional r139 candidates (not enabled)

GMM tuning_config values 1/2/4/8/10, split_item 2/3 and group_list_type 0/1 did not produce a useful improvement. A fused expert-major down gather saved about 6 us/layer in isolation but the direct-gather Cartesian path was faster in the selected comparison. The standalone probe.py needs gather.cpp built as libdown_gather.so beside it.

Real lm_head W8 chunk sizes 62080/31040/15520/65504 were compared with cached BF16. BF16 was around 1.00 ms versus 1.67 ms for W8 but had small logit bit differences. Raw BF16 chunks also differed, so none was enabled. Tests were on real head weights with 16 dynamic inputs. Argmax matching in these tests is not a general accuracy guarantee.

A 62080-column slice was dequantized through CANN with an identity input. The resulting weights and CPU BF16 weights were bit-identical, while both raw matmul outputs still differed from W8. Thus the observed slice difference comes from the accumulation path rather than the stored dequantized weights. This does not prove a universal all-input statement.

The production change retains the original head precision/implementation. No raw weights, private process environment, or credentials are included here.
