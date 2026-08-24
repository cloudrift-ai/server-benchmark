"""Execute experiment rows on provisioned or pre-allocated hosts."""

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path

from emmy.benchmark.bench_logging import _get_group_logger, active_run_dir, add_group_file_handler
from emmy.benchmark.command_workload import run_command_workload
from emmy.benchmark.experiment_record import ExperimentRecord, Infrastructure, Provenance
from emmy.benchmark.workload import capture_server_log, run_benchmark_workload
from emmy.deploy import DeployParams
from emmy.deploy import deploy as deploy_entry
from emmy.deploy import teardown as teardown_entry
from emmy.planner import BenchmarkTask, ExecutionGroup
from emmy.provisioning.cloud import delete_cloud_vm, provision_cloud_vm
from emmy.provisioning.host import RemoteHost
from emmy.provisioning.remote import provision_remote
from emmy.provisioning.ssh_transport import REMOTE_DEPLOY_DIR, make_run_cmd
from emmy.provisioning.staging import stage_to_remote
from emmy.redact import redact_secrets, register_secret
from emmy.system_info import SystemInformation
from emmy.timing import (
    PHASE_BENCHMARK,
    PHASE_COMMAND,
    PHASE_REMOTE_PROVISION,
    PHASE_TEARDOWN,
    PHASE_VM_PROVISION,
    PhaseTimer,
)


def _persist(task: BenchmarkTask, dry_run: bool) -> None:
    if not dry_run and task.record is not None:
        task.record.write(task.record_path())


def _ensure_records(group: ExecutionGroup, dry_run: bool) -> None:
    run_id = ExperimentRecord.new_run_id()
    for task in group.tasks:
        if task.record is None:
            task.record = ExperimentRecord.create(task, run_id)
        task.record.start("provisioning")
        _persist(task, dry_run)


def _infrastructure(group: ExecutionGroup, group_label: str, conn, *, preallocated: bool) -> Infrastructure:
    provider = None
    instance_id = None
    zone = None
    if conn.delete_info:
        provider = conn.delete_info[0]
        instance_id = conn.delete_info[1]
        if conn.delete_info[0] == "gcp" and len(conn.delete_info) > 2:
            zone = conn.delete_info[2]
    return Infrastructure(
        group=group_label,
        requested_gpu=group.gpu_name,
        requested_gpu_count=group.gpu_count,
        address=conn.address,
        ssh_port=conn.ssh_port,
        provider=provider,
        instance_id=instance_id,
        zone=zone,
        state="external" if preallocated else "active",
    )


def _finalize_failure(
    task: BenchmarkTask,
    *,
    stage: str,
    error: str,
    timing: dict[str, float],
    dry_run: bool,
) -> None:
    if task.record is None:
        return
    task.record.finish(success=False, stage=stage, timing=timing, error=error)
    _persist(task, dry_run)


async def run_execution_group(
    group: ExecutionGroup,
    config: dict,
    ssh_key: str,
    dry_run: bool = False,
    no_teardown: bool = False,
    preallocated_conn=None,
    provider: str | None = None,
) -> list[tuple[BenchmarkTask, bool, dict]]:
    """Run every experiment row in one host-sharing execution group."""
    task_results: list[tuple[BenchmarkTask, bool, dict]] = []
    completed: set[int] = set()
    task_timers: dict[int, PhaseTimer] = {}
    model_dir = config["benchmark"].get("model_dir", "/hf_models")
    hf_token = os.environ.get("HF_TOKEN", "")
    register_secret(hf_token)
    providers_config = config.get("providers", {})

    group_label = group.label
    logger = _get_group_logger(group)
    _ensure_records(group, dry_run)

    group_handler = None
    if not dry_run and group.tasks and group.tasks[0].run_dir is not None:
        group_handler = add_group_file_handler(group.tasks[0].run_dir, group_label)
        active_run_dir.set(group.tasks[0].run_dir)

    logger.info(f"Starting group: {group.gpu_name} x{group.gpu_count} ({len(group.tasks)} tasks)")
    conn = None
    infrastructure = None
    group_timer = PhaseTimer()
    try:
        if preallocated_conn is not None:
            conn = preallocated_conn
            logger.info(f"Using pre-allocated host: {conn.address}:{conn.ssh_port}")
        else:
            async with group_timer.ameasure(PHASE_VM_PROVISION):
                conn = await provision_cloud_vm(
                    group.gpu_name,
                    group.gpu_count,
                    ssh_key,
                    providers_config,
                    server_name=group_label,
                    dry_run=dry_run,
                    logger=logger,
                    provider=provider,
                )
            if conn is None:
                raise RuntimeError("VM provisioning failed")
            instance_id = f" (instance_id={conn.delete_info[1]})" if conn.delete_info else ""
            logger.info(f"VM provisioned: {conn.address}:{conn.ssh_port}{instance_id}")

        infrastructure = _infrastructure(group, group_label, conn, preallocated=preallocated_conn is not None)
        for task in group.tasks:
            task.record.execution.infrastructure = replace(infrastructure)
            _persist(task, dry_run)

        first_recipe = group.tasks[0].recipe if group.tasks else None
        host = RemoteHost(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
        async with group_timer.ameasure(PHASE_REMOTE_PROVISION):
            await provision_remote(
                host,
                driver_version=first_recipe.deploy.driver_version if first_recipe else None,
                cuda_version=first_recipe.deploy.cuda_version if first_recipe else None,
            )

        sysinfo_run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
        system = await SystemInformation.retrieve(sysinfo_run_cmd)
        for task in group.tasks:
            task.record.system = system
            task.record.execution.stage = "staging"
            _persist(task, dry_run)

        repo_dir_remote: str | None = None
        stage_paths: list[str] = []
        for task in group.tasks:
            if task.recipe.kind == "command" and task.recipe.command and task.recipe.command.stage:
                for path in task.recipe.command.stage:
                    if path not in stage_paths:
                        stage_paths.append(path)
        if stage_paths:
            repo_dir_remote = f"{REMOTE_DEPLOY_DIR}/{group_label}/repo"
            strict_stage = any(
                task.recipe.command.strict for task in group.tasks if task.recipe.kind == "command" and task.recipe.command is not None
            )
            staged_provenance = await stage_to_remote(
                Path.cwd(),
                stage_paths,
                conn.address,
                ssh_key,
                conn.ssh_port,
                repo_dir_remote,
                dry_run=dry_run,
                require_clean=strict_stage,
            )
            if staged_provenance is not None:
                for task in group.tasks:
                    if task.recipe.kind == "command" and task.recipe.command and task.recipe.command.stage:
                        task.record.provenance = Provenance(
                            git_revision=staged_provenance.git_revision,
                            git_dirty=staged_provenance.git_dirty,
                        )
                        _persist(task, dry_run)
        for task in group.tasks:
            active_run_dir.set(task.run_dir)
            recipe = task.recipe
            task_logger = _get_group_logger(group, task.model_name)
            task_logger.info(f"Recipe: {task.recipe_dir} (variant: {task.variant})")

            task_timer = PhaseTimer()
            task_timers[id(task)] = task_timer
            for phase_name, seconds in group_timer.phases.items():
                task_timer.record(phase_name, seconds, log=False)
            gpu_device_ids = list(range(task.gpu_count))
            task.record.execution.stage = "command" if recipe.kind == "command" else "deploy"
            _persist(task, dry_run)

            if recipe.kind == "command":
                run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
                run_id = task.record.execution.run_id
                task_dir_remote = f"{REMOTE_DEPLOY_DIR}/{group_label}/{task.variant}/{run_id}"
                command_info: dict = {"result_paths": [], "result_errors": []}
                try:
                    async with task_timer.ameasure(PHASE_COMMAND):
                        success, command_info = await run_command_workload(
                            task,
                            run_cmd,
                            repo_dir=repo_dir_remote,
                            task_dir=task_dir_remote,
                            gpu_device_ids=gpu_device_ids,
                            server=conn.address,
                            ssh_key=ssh_key,
                            ssh_port=conn.ssh_port,
                            dry_run=dry_run,
                        )
                except Exception as exc:
                    task_logger.error(f"Command workload error: {exc}")
                    success = False
                    command_info["error"] = str(exc)
                if recipe.command.strict and not dry_run:
                    if errors := task.record.missing_command_provenance():
                        command_info["provenance_errors"] = errors
                        success = False
                if errors := command_info.get("provenance_errors"):
                    task_logger.error("Required command provenance is missing: %s", ", ".join(errors))

                timing = task_timer.as_dict()
                error = command_info.get("error") or "; ".join(command_info.get("result_errors", [])) or None
                task.record.finish(success=success, stage="command", timing=timing, error=error)
                _persist(task, dry_run)
                task_results.append((task, success or dry_run, timing))
                completed.add(id(task))
                continue

            params = DeployParams(
                server=conn.address,
                ssh_key=ssh_key,
                ssh_port=conn.ssh_port,
                recipe=recipe,
                model_dir=model_dir,
                hf_token=hf_token,
                dry_run=dry_run,
                gpu_device_ids=gpu_device_ids,
                port_mappings=conn.port_mappings,
            )
            task_logger.info("Deploying model...")
            deployed = await deploy_entry(params, timer=task_timer, check_smoke_output=False)
            if not deployed:
                timing = task_timer.as_dict()
                task_logger.error("Deploy failed, skipping benchmark")
                _finalize_failure(
                    task,
                    stage="deploy",
                    error="model deployment failed",
                    timing=timing,
                    dry_run=dry_run,
                )
                task_results.append((task, False, timing))
                completed.add(id(task))
                continue

            task.record.execution.stage = "benchmark"
            _persist(task, dry_run)
            task_logger.info("Running benchmark...")
            run_cmd = make_run_cmd(conn.address, ssh_key, conn.ssh_port, dry_run=dry_run)
            async with task_timer.ameasure(PHASE_BENCHMARK):
                success, output, stderr, _bench_command = await run_benchmark_workload(run_cmd, recipe, dry_run=dry_run)

            benchmark_log = task.benchmark_log_path()
            if not dry_run:
                raw_output = "\n".join(part.rstrip() for part in (output, stderr) if part)
                benchmark_log.write_text(redact_secrets(raw_output) + ("\n" if raw_output else ""), encoding="utf-8")
            server_log_path = task.run_dir / f"{task.file_stem}.server.log"
            server_log = await capture_server_log(run_cmd, server_log_path, dry_run=dry_run)
            if server_log["status"] == "failed":
                task_logger.error("Failed to collect the raw server log")
                success = False

            if not no_teardown:
                task_logger.info("Tearing down...")
                async with task_timer.ameasure(PHASE_TEARDOWN):
                    await teardown_entry(params)

            timing = task_timer.as_dict()
            task.record.finish(success=success, stage="benchmark", timing=timing)
            _persist(task, dry_run)
            task_results.append((task, success or dry_run, timing))
            completed.add(id(task))

    except Exception as exc:
        logger.error(f"Execution group failed: {exc}")
        for task in group.tasks:
            if id(task) in completed:
                continue
            timer = task_timers.get(id(task))
            timing = timer.as_dict() if timer is not None else group_timer.as_dict()
            stage = task.record.execution.stage if task.record else "execution"
            _finalize_failure(
                task,
                stage=stage,
                error=str(exc),
                timing=timing,
                dry_run=dry_run,
            )
            task_results.append((task, False, timing))
            completed.add(id(task))
    finally:
        active_run_dir.set(None)
        cleanup_error = None
        infrastructure_state = None
        if preallocated_conn is not None and conn is not None:
            infrastructure_state = "external"
            logger.info(f"Leaving pre-allocated host in place: {conn.address}")
        elif conn is not None and conn.delete_info:
            if no_teardown:
                infrastructure_state = "active"
                logger.info(f"Skipping VM deletion (--no-teardown): {conn.address}")
            else:
                logger.info("Deleting VM...")
                try:
                    deleted = await delete_cloud_vm(conn.delete_info, dry_run)
                    if deleted is False:
                        raise RuntimeError("provider reported that VM deletion failed")
                    infrastructure_state = "deleted"
                    logger.info("VM deleted.")
                except Exception as exc:
                    infrastructure_state = "delete_failed"
                    cleanup_error = str(exc)
                    logger.error(f"Failed to delete VM: {exc}")

        if infrastructure_state is not None:
            for task in group.tasks:
                if task.record is None or task.record.execution.infrastructure is None:
                    continue
                task.record.execution.infrastructure.state = infrastructure_state
                if cleanup_error:
                    task.record.execution.cleanup_error = cleanup_error
                    if task.record.status == "succeeded":
                        task.record.finish(
                            success=False,
                            stage="vm_cleanup",
                            timing=task.record.execution.timing_seconds,
                            error=cleanup_error,
                        )
                _persist(task, dry_run)
            if cleanup_error:
                task_results = [(task, False, timing) for task, _ok, timing in task_results]

        if group_handler is not None:
            logging.getLogger().removeHandler(group_handler)
            group_handler.close()

    logger.info(f"Completed group: {group_label}")
    return task_results


async def _run_groups_on_hosts(groups, hosts: list, config, ssh_key, dry_run, provider: str | None = None):
    """Dispatch groups across a fixed pool of compatible hosts."""
    locks: dict[int, asyncio.Lock] = {id(host): asyncio.Lock() for host in hosts}
    select_lock = asyncio.Lock()
    in_use: set[int] = set()

    async def _acquire_host(group):
        while True:
            async with select_lock:
                for host in hosts:
                    if id(host) in in_use:
                        continue
                    if dry_run or host.satisfies(group.gpu_name, group.gpu_count):
                        in_use.add(id(host))
                        return host
            await asyncio.sleep(0.05)

    async def _run_one(group):
        host = await _acquire_host(group)
        try:
            async with locks[id(host)]:
                return await run_execution_group(
                    group,
                    config,
                    ssh_key,
                    dry_run,
                    no_teardown=True,
                    preallocated_conn=host.conn,
                    provider=provider,
                )
        finally:
            in_use.discard(id(host))

    return await asyncio.gather(*(_run_one(group) for group in groups), return_exceptions=True)


async def _run_groups(groups, config, ssh_key, dry_run, max_workers, no_teardown=False, provider: str | None = None):
    """Run execution groups concurrently with a semaphore."""
    sem = asyncio.Semaphore(max_workers or len(groups))

    async def _run_with_semaphore(group):
        async with sem:
            return await run_execution_group(group, config, ssh_key, dry_run, no_teardown, provider=provider)

    return await asyncio.gather(*(_run_with_semaphore(group) for group in groups), return_exceptions=True)
