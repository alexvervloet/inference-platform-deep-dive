# Lessons learned

## Capacity boundary fixtures must show their arithmetic

- Expected: the unsharded-KV test case would exceed a 1 GiB accelerator while the sharded case fit.
- Actual: its weights used 0.5 GiB and two KV reservations used another 0.5 GiB, landing exactly on the inclusive fit boundary.
- Next time: calculate and state both sides of a capacity boundary when constructing the fixture; make the counterfactual cross the boundary by a visible margin rather than relying on mental arithmetic.

## A control test must prove its setup reached the intended boundary

- Expected: an “active” request would fill live-token capacity, allowing the next request to exercise the bounded queue.
- Actual: that setup request exceeded the per-request limit and was shed by an earlier control, so the next request was correctly admitted into otherwise empty capacity.
- Next time: assert the setup decision before asserting a later boundary, and choose fixtures that visibly satisfy every earlier precondition in the decision order.

## Assert decision evidence at the right collection level

- Expected: a substring assertion would confirm that the TPOT gate caused rollback.
- Actual: `assertIn` searched for an abbreviated string as an exact tuple member, even though the tuple correctly contained the full deciding message.
- Next time: use exact membership for stable control codes/messages, or inspect a specific message when intentionally checking a substring; do not blur the two forms.

## A fresh Python 3.13 venv may not contain the declared build backend

- Expected: an editable install with `--no-build-isolation` would verify packaging without network access.
- Actual: the fresh environment contained `pip` but no `setuptools`, so it could not import `setuptools.build_meta`; the project correctly declares `setuptools>=75`, but fulfilling that declaration needs the normal isolated installer or a pre-provisioned build tool.
- Next time: distinguish an offline source-tree test (`PYTHONPATH=.`) from a clean installation test; use the documented normal install when network/package mirrors are available, and do not claim `--no-build-isolation` works in a bare environment.
