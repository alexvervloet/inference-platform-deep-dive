# Lessons learned

## Capacity boundary fixtures must show their arithmetic

- Expected: the unsharded-KV test case would exceed a 1 GiB accelerator while the sharded case fit.
- Actual: its weights used 0.5 GiB and two KV reservations used another 0.5 GiB, landing exactly on the inclusive fit boundary.
- Next time: calculate and state both sides of a capacity boundary when constructing the fixture; make the counterfactual cross the boundary by a visible margin rather than relying on mental arithmetic.

## A control test must prove its setup reached the intended boundary

- Expected: an "active" request would fill live-token capacity, allowing the next request to exercise the bounded queue.
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

## A cited algorithm's token accounting has to be read, not remembered

- Expected: a speculative round emits the accepted draft prefix, plus one correction when the draft diverges.
- Actual: the verification pass covers every draft position *plus the one after it*, so a fully accepted round of `k` drafts emits `k + 1` tokens. Special-casing full acceptance to emit `k` understated the speedup exactly where speculation pays off, in a lesson that cites the paper defining the round.
- Next time: when an example implements a published algorithm, write the step count from the paper's own procedure into a comment beside the code, so a later reader checks the arithmetic against the source rather than against intuition.

## A rounded metric ties, and the tie-break names a winner nobody chose

- Expected: taking the maximum of prefill, decode, concurrency, and floor requirements would name the dimension that binds fleet size.
- Actual: `ceil()` to whole replicas put all four on 2, and the alphabetical tie-break reported "request concurrency", the dimension with the *most* slack (1.08 raw against decode's 1.77). The capstone's headline reason was an artifact of sorting.
- Next time: when a decision takes a max over rounded values, return every dimension that reaches the maximum. If a single name is still wanted, rank it by the unrounded value, and treat "only the floor binds" as a result worth showing rather than hiding.

## The capstone can contradict the chapter it demonstrates

- Expected: the integrated plan would obey 22.2, which exists to say that weight fit is not service fit.
- Actual: the capstone described the model twice, once for the memory assessment and once inline for the parallel planner, and handed placement a per-GPU figure covering weights and a runtime constant but not the KV reservation it had just computed. Nothing failed, because the fixture had slack.
- Next time: derive every downstream input from the upstream decision object instead of restating it, and add a counterfactual that sits between the two numbers, here inventory that fits weights but not KV, so the omission cannot pass silently.

## A field named for a span can hold an absolute timestamp

- Expected: `makespan_steps` measured the length of the simulated run.
- Actual: it returned the final step counter. Every fixture arrived at step 0, so it read correctly, and the exercise that asks learners to add arrival streams is exactly where it would stop.
- Next time: test a time-valued result under a shifted origin. A quantity that claims to be a duration should be invariant when every input timestamp moves by the same amount.
