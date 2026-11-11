# r140 calibration alternatives

HC activation modes 1/2/3 use reciprocal or multiplication reassociation. They are not selected: exhaustive finite BF16 gate testing found 1048 differences for reciprocal modes, and the SiLU reassociation had one or 901 differences depending on mode. Mode0 uses the original division and intermediate BF16 rounding and passed.

The production wrapper hardcodes mode0. The calibration retains alternatives to make the precision trap reproducible. Multiplying a sigmoid, or omitting an intermediate BF16 cast, must not be assumed bit-equivalent to the original operations.

For FP32 normalization Mul+Cast, the isolated benefit was small and is not reported as a significant independent speedup. Request-level A/B results decide whether to deploy the overall bundle.
