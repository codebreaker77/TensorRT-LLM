# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest

import tensorrt_llm.executor.executor as executor_module
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.postproc_worker import PostprocWorkerConfig
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

pytestmark = pytest.mark.cpu_only
_CREATE_IPC_EXECUTOR = GenerationExecutor._create_ipc_executor


@pytest.fixture
def factories(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    monkeypatch.setattr(executor_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(executor_module, "external_mpi_comm_available", lambda _: False)
    monkeypatch.setattr(executor_module, "need_spawn_mpi_workers", lambda _: False)
    monkeypatch.setattr(executor_module, "enable_worker_single_process_for_tp1", lambda: False)
    result = {name: Mock() for name in ("ipc", "rpc", "ray")}
    for name, factory in result.items():
        monkeypatch.setattr(GenerationExecutor, f"_create_{name}_executor", factory)
    return result


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_default_without_mpi_uses_in_process_worker(
    monkeypatch: pytest.MonkeyPatch, factories: dict[str, Mock], system: str
) -> None:
    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", False)
    monkeypatch.setattr(executor_module.platform, "system", lambda: system)
    process_pool = Mock(side_effect=AssertionError("must not create a process pool"))
    monkeypatch.setattr(executor_module, "ProcessPoolExecutorSession", process_pool)

    result = GenerationExecutor.create(world_size=1)

    assert result is factories["ipc"].return_value
    factories["ipc"].assert_called_once()
    assert factories["ipc"].call_args.kwargs["use_worker"] is True
    assert factories["ipc"].call_args.kwargs["mpi_session"] is None
    process_pool.assert_not_called()
    factories["rpc"].assert_not_called()
    factories["ray"].assert_not_called()


@pytest.mark.parametrize("model_world_size", [2, 4])
def test_without_mpi_rejects_multiple_in_process_ranks(
    monkeypatch: pytest.MonkeyPatch, factories: dict[str, Mock], model_world_size: int
) -> None:
    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", False)

    with pytest.raises(ValueError, match="ENABLE_MULTI_DEVICE.*world_size=1"):
        GenerationExecutor.create(world_size=1, model_world_size=model_world_size)

    for factory in factories.values():
        factory.assert_not_called()


def test_default_with_mpi_keeps_proxy(
    monkeypatch: pytest.MonkeyPatch, factories: dict[str, Mock]
) -> None:
    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", True)

    GenerationExecutor.create(world_size=1)

    assert factories["ipc"].call_args.kwargs["use_worker"] is False


@pytest.mark.parametrize("num_postprocess_workers", [0, 2])
def test_without_mpi_constructs_worker_with_postprocessing_config(
    monkeypatch: pytest.MonkeyPatch, factories: dict[str, Mock], num_postprocess_workers: int
) -> None:
    from tensorrt_llm.executor import worker as worker_module

    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", False)
    monkeypatch.setattr(
        GenerationExecutor, "_create_ipc_executor", staticmethod(_CREATE_IPC_EXECUTOR)
    )
    worker = Mock()
    monkeypatch.setattr(worker_module, "GenerationExecutorWorker", worker)
    config = PostprocWorkerConfig(num_postprocess_workers=num_postprocess_workers)
    model_dir = Path("model")
    tokenizer = Mock()

    result = GenerationExecutor.create(
        world_size=1,
        postproc_worker_config=config,
        is_llm_executor=True,
        hf_model_dir=model_dir,
        tokenizer=tokenizer,
    )

    assert result is worker.return_value
    worker.assert_called_once_with(
        hf_model_dir=model_dir,
        tokenizer=tokenizer,
        llm_args=None,
        postproc_worker_config=config,
        is_llm_executor=True,
    )
    factories["rpc"].assert_not_called()
    factories["ray"].assert_not_called()


@pytest.mark.parametrize("enable_multi_device", [False, True])
@pytest.mark.parametrize("selection", ["return_logits", "environment"])
def test_explicit_in_process_selection(
    monkeypatch: pytest.MonkeyPatch,
    factories: dict[str, Mock],
    enable_multi_device: bool,
    selection: str,
) -> None:
    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", enable_multi_device)
    monkeypatch.setattr(
        executor_module, "enable_worker_single_process_for_tp1", lambda: selection == "environment"
    )

    GenerationExecutor.create(world_size=1, return_logits=selection == "return_logits")

    assert factories["ipc"].call_args.kwargs["use_worker"] is True


@pytest.mark.parametrize("enable_multi_device", [False, True])
def test_external_session_is_preserved(
    monkeypatch: pytest.MonkeyPatch, factories: dict[str, Mock], enable_multi_device: bool
) -> None:
    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", enable_multi_device)
    session = Mock()

    GenerationExecutor.create(world_size=1, mpi_session=session)

    assert factories["ipc"].call_args.kwargs["use_worker"] is False
    assert factories["ipc"].call_args.kwargs["mpi_session"] is session


@pytest.mark.parametrize("enable_multi_device", [False, True])
@pytest.mark.parametrize("orchestrator", ["ray", "rpc"])
def test_explicit_orchestrator_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    factories: dict[str, Mock],
    enable_multi_device: bool,
    orchestrator: str,
) -> None:
    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", enable_multi_device)
    llm_args = TorchLlmArgs.model_construct(orchestrator_type=orchestrator)

    result = GenerationExecutor.create(world_size=1, llm_args=llm_args)

    assert result is factories[orchestrator].return_value
    factories[orchestrator].assert_called_once()
    factories["ipc"].assert_not_called()


@pytest.mark.parametrize("launch", ["spawn", "external"])
def test_multi_device_launch_is_preserved(
    monkeypatch: pytest.MonkeyPatch, factories: dict[str, Mock], launch: str
) -> None:
    monkeypatch.setattr(executor_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(executor_module, "need_spawn_mpi_workers", lambda _: launch == "spawn")
    monkeypatch.setattr(
        executor_module, "external_mpi_comm_available", lambda _: launch == "external"
    )
    session = Mock() if launch == "external" else None

    GenerationExecutor.create(
        world_size=1, model_world_size=2, mpi_session=session, reuse_mpi_comm=launch == "external"
    )

    assert factories["ipc"].call_args.kwargs["use_worker"] is False
    assert factories["ipc"].call_args.kwargs["model_world_size"] == 2
    assert factories["ipc"].call_args.kwargs["mpi_session"] is session
