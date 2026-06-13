"""Tests for the request scheduler.

Runs with pytest if available, or standalone: ``python test/test_scheduler.py``.
No third-party test deps required.

The scheduler grants slots on later event-loop ticks, so the tests use a small
helper: fill every slot, queue several waiters, then release slots one at a time
and record the order in which queued waiters wake up.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scheduler import Scheduler


async def _drain_order(policy, jobs, max_concurrent=1):
    """Return the order waiters acquire slots under ``policy``.

    Every slot is pre-filled with a blocker first, so all ``jobs`` are forced to
    *queue* — otherwise the first arrival would grab a free slot before any
    ordering applies. We then release the blockers; each freed slot cascades to
    the best-ranked waiter, which records itself and releases in turn.
    """
    sched = Scheduler(max_concurrent, policy=policy)
    order = []

    for _ in range(max_concurrent):       # occupy every slot
        await sched.acquire()

    async def worker(label, priority, cost_hint):
        await sched.acquire(priority=priority, cost_hint=cost_hint)
        order.append(label)
        await sched.release()             # hand the slot to the next-best waiter

    tasks = []
    for label, priority, cost_hint in jobs:
        tasks.append(asyncio.ensure_future(worker(label, priority, cost_hint)))
        await asyncio.sleep(0)            # reach acquire() and enqueue in arrival order

    for _ in range(max_concurrent):       # release blockers; queue drains by policy
        await sched.release()

    await asyncio.gather(*tasks)
    return order


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_rejects_unknown_policy():
    try:
        Scheduler(1, policy="lifo")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown policy")


def test_fifo_preserves_arrival_order():
    jobs = [("a", 0, 100), ("b", 0, 10), ("c", 0, 50)]
    order = _run(_drain_order("fifo", jobs))
    assert order == ["a", "b", "c"], order


def test_priority_orders_high_first_ties_by_arrival():
    # priorities: a=1, b=5, c=5, d=0 -> high first, ties (b,c) by arrival: b,c,a,d
    jobs = [("a", 1, 0), ("b", 5, 0), ("c", 5, 0), ("d", 0, 0)]
    order = _run(_drain_order("priority", jobs))
    assert order == ["b", "c", "a", "d"], order


def test_sjf_orders_shortest_job_first():
    # cost hints (max_tokens): a=300, b=20, c=120 -> expect b, c, a
    jobs = [("a", 0, 300), ("b", 0, 20), ("c", 0, 120)]
    order = _run(_drain_order("sjf", jobs))
    assert order == ["b", "c", "a"], order


def test_concurrency_cap_is_respected():
    async def scenario():
        sched = Scheduler(2, policy="fifo")
        await sched.acquire()
        await sched.acquire()
        assert sched.active == 2
        # third acquire must park, not exceed the cap
        third = asyncio.ensure_future(sched.acquire())
        await asyncio.sleep(0)
        assert sched.active == 2
        assert sched.waiting() == 1
        await sched.release()
        await asyncio.sleep(0)
        await third  # the parked waiter now holds the freed slot
        assert sched.active == 2
        assert sched.waiting() == 0
    _run(scenario())


def test_cancelled_waiter_does_not_leak_a_slot():
    async def scenario():
        sched = Scheduler(1, policy="fifo")
        await sched.acquire()              # slot full
        waiter = asyncio.ensure_future(sched.acquire())
        await asyncio.sleep(0)
        assert sched.waiting() == 1
        waiter.cancel()                    # client disconnects while queued
        await asyncio.sleep(0)
        # releasing the only active slot should leave it free, not double-counted
        await sched.release()
        await asyncio.sleep(0)
        assert sched.active == 0
        # a fresh request can still acquire immediately
        await asyncio.wait_for(sched.acquire(), timeout=1.0)
        assert sched.active == 1
    _run(scenario())


def _run_standalone():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
