# Initial experiment record — 2026-11-20

## Environment and commands

Local Python, standard library only:

```bash
python -m unittest discover -s tests -v
python demo.py
```

The test suite includes an **independent imperative execution oracle** for feasible cycle counts. The main trillion-cycle example is **not** replayed. Instead the exact semantics of the supported Ground family are cross-checked on many small cases and the composed far-horizon algebra is checked for self-consistency.

## Results

- 14 unit tests passed on the development machine, including 1-byte exhaustive tests over all 256 values for each primary unary opcode and randomized multi-byte program/coordinate comparisons.
- Request coordinate: `1,000,000,000,123` program cycles.
- Byte-processor program: 6 instructions per cycle; direct replay would require `6,000,000,000,738` instruction executions.
- Exact Ground composition: 58 products (different cost units; not an apples-to-apples wall-clock acceleration ratio).
- Desired byte 0: `0xA5` and high nibble of byte 3: `0xB`.
- Constructed input: `d20000a0`; requested output: `a5b3`, satisfying both constraints.
- Free input bits: 20, so the intention does not uniquely determine the initial memory. The solver chooses a canonical valid candidate.
- Long-coordinate recheck status: deterministic same-Ground recomputation for trillion-cycle result; separately implemented sequential oracle conformance only for feasible short runs.
- Observability study: one-bit per-time observation of a rotating byte gives ranks 1 through 8; the latent byte is exactly reconstructed from eight time samples. A zero-setting program produces rank zero under repeated later observations.

These are *mathematical and software conformance observations*, not an independent Lean verification, distributed host observation, or a claim of general CPU superiority.

