# Lessons learned

## Capacity boundary fixtures must show their arithmetic

- Expected: the unsharded-KV test case would exceed a 1 GiB accelerator while the sharded case fit.
- Actual: its weights used 0.5 GiB and two KV reservations used another 0.5 GiB, landing exactly on the inclusive fit boundary.
- Next time: calculate and state both sides of a capacity boundary when constructing the fixture; make the counterfactual cross the boundary by a visible margin rather than relying on mental arithmetic.

## A control test must prove its setup reached the intended boundary

- Expected: an “active” request would fill live-token capacity, allowing the next request to exercise the bounded queue.
- Actual: that setup request exceeded the per-request limit and was shed by an earlier control, so the next request was correctly admitted into otherwise empty capacity.
- Next time: assert the setup decision before asserting a later boundary, and choose fixtures that visibly satisfy every earlier precondition in the decision order.
