# Lessons learned

## Capacity boundary fixtures must show their arithmetic

- Expected: the unsharded-KV test case would exceed a 1 GiB accelerator while the sharded case fit.
- Actual: its weights used 0.5 GiB and two KV reservations used another 0.5 GiB, landing exactly on the inclusive fit boundary.
- Next time: calculate and state both sides of a capacity boundary when constructing the fixture; make the counterfactual cross the boundary by a visible margin rather than relying on mental arithmetic.
