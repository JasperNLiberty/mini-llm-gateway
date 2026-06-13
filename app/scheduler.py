"""Priority-aware request scheduler for the gateway.

Replaces a plain FIFO semaphore with a pluggable admission policy, so the order
in which queued requests acquire a concurrency slot becomes a *decision* rather
than just arrival order. This is where head-of-line blocking is won or lost:
under load, a 30-token request stuck behind a 4k-token generation waits for the
whole thing to finish unless the scheduler is allowed to reorder the queue.

Concurrency is still capped at ``max_concurrent`` (that cap is the backpressure
mechanism); the scheduler only governs *which* waiter gets the next freed slot.

Policies (``SCHED_POLICY`` env var):

  ``fifo``      first-come-first-served. Default; identical ordering to the
                semaphore it replaces.
  ``priority``  higher request ``priority`` first; ties broken by arrival order.
                Lets you put interactive traffic ahead of batch/background work.
  ``sjf``       shortest job first, by ``max_tokens`` hint. Minimizes mean wait
                time and directly attacks head-of-line blocking — at the cost of
                potentially starving large jobs under sustained load.

The implementation is a min-heap of waiters, each parked on an ``asyncio.Future``
that the scheduler resolves when it grants a slot. All state transitions happen
under a single lock, so grant/release ordering is deterministic.
"""

import asyncio
import heapq
import itertools
import os
from contextlib import asynccontextmanager

POLICIES = ("fifo", "priority", "sjf")


class _Waiter:
    __slots__ = ("future", "granted")

    def __init__(self, future: "asyncio.Future") -> None:
        self.future = future
        self.granted = False


class Scheduler:
    """Admission scheduler capping concurrency and ordering the wait queue.

    Use via the async context manager::

        async with scheduler.slot(priority=p, cost_hint=max_tokens):
            ...  # holds one concurrency slot for the duration
    """

    def __init__(self, max_concurrent: int, policy: str = "fifo") -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown SCHED_POLICY {policy!r}; choose one of {POLICIES}")
        self.max_concurrent = max_concurrent
        self.policy = policy
        self._active = 0
        self._heap: list = []
        self._seq = itertools.count()
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        """Slots currently in use."""
        return self._active

    def waiting(self) -> int:
        """Requests parked in the queue waiting for a slot."""
        return len(self._heap)

    def _key(self, priority: int, cost_hint: int, seq: int) -> tuple:
        # ``seq`` is unique and present in every key, so heap comparisons never
        # reach the (incomparable) _Waiter payload — it is a guaranteed tiebreak.
        if self.policy == "priority":
            return (-priority, seq)
        if self.policy == "sjf":
            return (cost_hint, seq)
        return (seq,)

    def _pump(self) -> None:
        """Grant slots to the best waiters until the cap is hit or queue drains.

        Caller must hold ``self._lock``. ``set_result`` only schedules the parked
        coroutine to resume on a later tick, so calling it under the lock is safe.
        """
        while self._heap and self._active < self.max_concurrent:
            _, _, waiter = heapq.heappop(self._heap)
            if waiter.future.done():  # cancelled while it sat in the queue
                continue
            self._active += 1
            waiter.granted = True
            waiter.future.set_result(None)

    async def acquire(self, priority: int = 0, cost_hint: int = 0) -> None:
        waiter = _Waiter(asyncio.get_event_loop().create_future())
        async with self._lock:
            seq = next(self._seq)
            heapq.heappush(self._heap, (self._key(priority, cost_hint, seq), seq, waiter))
            self._pump()
        try:
            await waiter.future
        except asyncio.CancelledError:
            # If a slot was granted in the same tick the awaiting task was
            # cancelled (e.g. client disconnect), hand it back so it isn't leaked.
            async with self._lock:
                if waiter.granted:
                    self._active -= 1
                    self._pump()
            raise

    async def release(self) -> None:
        async with self._lock:
            self._active -= 1
            self._pump()

    @asynccontextmanager
    async def slot(self, priority: int = 0, cost_hint: int = 0):
        await self.acquire(priority, cost_hint)
        try:
            yield
        finally:
            await self.release()


def from_env(max_concurrent: int) -> "Scheduler":
    """Build a Scheduler with policy taken from ``SCHED_POLICY`` (default fifo)."""
    return Scheduler(max_concurrent, policy=os.getenv("SCHED_POLICY", "fifo"))
