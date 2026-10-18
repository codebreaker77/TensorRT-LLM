# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest

from tensorrt_llm.serve.cluster_storage import Etcd3ClusterStorage

pytestmark = [pytest.mark.cpu_only, pytest.mark.asyncio]


@pytest.fixture
def storage(monkeypatch):
    client = Mock()
    client.lease.side_effect = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    client.put_if_not_exists.return_value = True
    monkeypatch.setattr(Etcd3ClusterStorage, "_connect", staticmethod(lambda _: client))
    return Etcd3ClusterStorage("etcd://localhost:2379", "test")


def rpc_error(code):
    error = grpc.RpcError()
    error.code = lambda: code
    return error


@pytest.mark.parametrize("ttl", [0, -1])
async def test_expired_refresh_allows_reregistration(storage, ttl):
    assert await storage.set("worker", "old", ttl=2)
    storage.client.refresh_lease.return_value = iter([SimpleNamespace(TTL=ttl)])

    assert not await storage.expire("worker", 2)
    assert await storage.set("worker", "new", ttl=2)
    assert storage.client.put_if_not_exists.call_args.kwargs["lease"].id == 2


async def test_empty_refresh_does_not_report_success(storage):
    assert await storage.set("worker", "old", ttl=2)
    storage.client.refresh_lease.return_value = iter([])
    assert not await storage.expire("worker", 2)


async def test_live_refresh_preserves_lease(storage):
    assert await storage.set("worker", "old", ttl=2)
    storage.client.refresh_lease.return_value = iter([SimpleNamespace(TTL=2)])
    assert await storage.expire("worker", 2)
    assert await storage.set("worker", "new", overwrite_if_exists=True, ttl=2)
    assert storage.client.put.call_args.kwargs["lease"].id == 1
    storage.client.lease.assert_called_once_with(2)


async def test_uncached_refresh_does_not_create_lease(storage):
    assert not await storage.expire("missing", 2)
    storage.client.lease.assert_not_called()
    storage.client.refresh_lease.assert_not_called()


async def test_refresh_not_found_allows_reregistration(storage):
    assert await storage.set("worker", "old", ttl=2)
    storage.client.refresh_lease.side_effect = rpc_error(grpc.StatusCode.NOT_FOUND)
    assert not await storage.expire("worker", 2)
    assert await storage.set("worker", "new", ttl=2)
    assert storage.client.put_if_not_exists.call_args.kwargs["lease"].id == 2


@pytest.mark.parametrize("overwrite", [False, True])
async def test_set_retries_expired_lease(storage, overwrite):
    assert await storage.set("worker", "old", ttl=2)
    put = storage.client.put if overwrite else storage.client.put_if_not_exists
    put.reset_mock()
    put.side_effect = [rpc_error(grpc.StatusCode.NOT_FOUND), True]

    assert await storage.set("worker", "new", overwrite_if_exists=overwrite, ttl=2)
    assert [call.kwargs["lease"].id for call in put.call_args_list] == [1, 2]


async def test_retry_preserves_existing_key(storage):
    assert await storage.set("worker", "old", ttl=2)
    storage.client.put_if_not_exists.side_effect = [rpc_error(grpc.StatusCode.NOT_FOUND), False]
    assert not await storage.set("worker", "new", ttl=2)
    assert storage.client.put_if_not_exists.call_count == 3
    storage.client.put.assert_not_called()


async def test_set_retry_is_bounded(storage):
    storage.client.put.side_effect = rpc_error(grpc.StatusCode.NOT_FOUND)
    assert not await storage.set("worker", "new", overwrite_if_exists=True, ttl=2)
    assert storage.client.put.call_count == 2
    assert "worker" not in storage._leases


@pytest.mark.parametrize("operation", ["set", "expire"])
async def test_unrelated_rpc_error_preserves_lease(storage, operation):
    assert await storage.set("worker", "old", ttl=2)
    error = rpc_error(grpc.StatusCode.PERMISSION_DENIED)
    if operation == "set":
        storage.client.put.side_effect = error
        call = storage.set("worker", "new", overwrite_if_exists=True, ttl=2)
    else:
        storage.client.refresh_lease.side_effect = error
        call = storage.expire("worker", 2)
    with pytest.raises(grpc.RpcError):
        await call
    assert storage._leases["worker"].id == 1
    storage.client.lease.assert_called_once_with(2)


async def test_persistent_set_does_not_retry_not_found(storage):
    storage.client.put.side_effect = rpc_error(grpc.StatusCode.NOT_FOUND)
    with pytest.raises(grpc.RpcError):
        await storage.set("worker", "new", overwrite_if_exists=True)
    storage.client.put.assert_called_once_with("worker", "new", lease=None)
    storage.client.lease.assert_not_called()


@pytest.mark.parametrize("ttl", [0, -1])
async def test_invalid_refresh_ttl(storage, ttl):
    assert not await storage.expire("worker", ttl)
    storage.client.lease.assert_not_called()
    storage.client.refresh_lease.assert_not_called()
