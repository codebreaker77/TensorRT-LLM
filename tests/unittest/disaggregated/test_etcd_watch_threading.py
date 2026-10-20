# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Etcd callbacks enqueue watch events on the queue's owning event loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.serve.cluster_storage import Etcd3WatchEventQueue, WatchEventType

pytestmark = pytest.mark.cpu_only


class PutEvent:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class DeleteEvent(PutEvent):
    pass


def _response():
    return SimpleNamespace(
        events=[PutEvent(b"workers/one", b"registered"), DeleteEvent(b"workers/two", b"")]
    )


def _deliver_from_thread(queue, response):
    with ThreadPoolExecutor(max_workers=1) as producer:
        producer.submit(queue.add_events_from_resp, response).result(timeout=2)


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("waiting_reader", [False, True])
def test_threaded_batch_is_enqueued_on_owner_loop(monkeypatch, debug, waiting_reader):
    async def exercise():
        cancel = Mock()
        queue = Etcd3WatchEventQueue("workers/", cancel)
        owner = get_ident()
        put_threads = []
        put_nowait = queue.events.put_nowait

        def checked_put(event):
            put_threads.append(get_ident())
            # Reject unsafe writes before they strand a waiting future, keeping
            # cleanup bounded even when testing a broken implementation.
            if get_ident() != owner:
                raise RuntimeError("Watch queue accessed outside its event loop")
            put_nowait(event)

        monkeypatch.setattr(queue.events, "put_nowait", checked_put)
        reader = asyncio.create_task(queue.drain()) if waiting_reader else None
        try:
            await asyncio.sleep(0)
            _deliver_from_thread(queue, _response())
            await asyncio.sleep(0)
            assert put_threads == [owner, owner]
            cancel.assert_not_called()
            if reader is None:
                reader = asyncio.create_task(queue.drain())
            events = await asyncio.wait_for(reader, 2)
            assert [(e.storage_item.key, e.storage_item.value, e.event_type) for e in events] == [
                ("workers/one", "registered", WatchEventType.SET),
                ("workers/two", "", WatchEventType.DELETE),
            ]
            assert queue.events.empty()
        finally:
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)

    asyncio.run(exercise(), debug=debug)


def test_cancelled_reader_does_not_consume_later_threaded_events():
    async def exercise():
        cancel = Mock()
        queue = Etcd3WatchEventQueue("workers/", cancel)
        reader = asyncio.create_task(queue.drain())
        await asyncio.sleep(0)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        _deliver_from_thread(queue, _response())
        events = await asyncio.wait_for(queue.drain(), 2)
        assert [e.storage_item.key for e in events] == ["workers/one", "workers/two"]
        cancel.assert_not_called()

    asyncio.run(exercise(), debug=True)


def test_callback_on_event_loop_delivers_events():
    async def exercise():
        cancel = Mock()
        queue = Etcd3WatchEventQueue("workers/", cancel)
        queue.add_events_from_resp(_response())
        events = await asyncio.wait_for(queue.drain(), 2)
        assert len(events) == 2
        cancel.assert_not_called()

    asyncio.run(exercise(), debug=True)


def test_empty_response_does_not_cancel_watch():
    async def exercise():
        cancel = Mock()
        queue = Etcd3WatchEventQueue("workers/", cancel)
        _deliver_from_thread(queue, SimpleNamespace(events=[]))
        await asyncio.sleep(0)
        assert queue.events.empty()
        cancel.assert_not_called()

    asyncio.run(exercise(), debug=True)


def test_invalid_response_cancels_watch():
    async def exercise():
        cancel = Mock()
        queue = Etcd3WatchEventQueue("workers/", cancel)
        _deliver_from_thread(queue, RuntimeError("watch connection failed"))
        assert queue.events.empty()
        cancel.assert_called_once_with()

    asyncio.run(exercise(), debug=True)


def test_late_callback_after_loop_shutdown_cancels_watch():
    cancel = Mock()

    async def make_queue():
        return Etcd3WatchEventQueue("workers/", cancel)

    queue = asyncio.run(make_queue(), debug=True)
    _deliver_from_thread(queue, _response())
    assert queue.events.empty()
    cancel.assert_called_once_with()
