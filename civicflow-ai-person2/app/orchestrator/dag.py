"""
CivicFlow AI - Asynchronous DAG Engine
========================================

Lightweight, dependency-aware DAG execution engine supporting:
- Async handler functions and parallel node execution via `asyncio.gather`
- Topological dependency resolution & cycle detection (Kahn's algorithm)
- Conditional execution predicates
- Shared context passing
- Graceful failure handling and fallback mechanisms
"""

import asyncio
from dataclasses import dataclass, field
import inspect
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from app.models.orchestrator import PipelineContext

logger = logging.getLogger(__name__)


@dataclass
class DAGNode:
    """Represents an executable node within the DAG graph."""

    node_id: str
    handler: Callable[[PipelineContext], Awaitable[Any]]
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[Callable[[PipelineContext], bool]] = None
    fallback: Optional[Callable[[PipelineContext, Exception], Awaitable[Any]]] = None
    retry_count: int = 0


class DAGOrchestrator:
    """Asynchronous DAG orchestrator for multi-agent pipeline execution."""

    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}

    def add_node(self, node: DAGNode) -> "DAGOrchestrator":
        """Register a new node into the DAG graph."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node ID '{node.node_id}' is already registered in DAG.")
        self.nodes[node.node_id] = node
        return self

    def validate_dag(self) -> List[str]:
        """Validates dependency existence and checks for cycles using Kahn's algorithm.

        Returns topological ordering list of node IDs.
        Raises ValueError if cycle or missing dependency detected.
        """
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        graph: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}

        # Build adjacency list & count in-degrees
        for node_id, node in self.nodes.items():
            for parent_id in node.depends_on:
                if parent_id not in self.nodes:
                    raise ValueError(
                        f"Node '{node_id}' depends on non-existent node '{parent_id}'."
                    )
                graph[parent_id].append(node_id)
                in_degree[node_id] += 1

        # Queue nodes with in-degree 0
        queue = [node_id for node_id, count in in_degree.items() if count == 0]
        topological_order: List[str] = []

        while queue:
            current = queue.pop(0)
            topological_order.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topological_order) != len(self.nodes):
            raise ValueError("Cycle detected in DAG definition. Graph must be acyclic.")

        return topological_order

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Executes the DAG asynchronously, running ready independent nodes in parallel."""
        # 1. Validate graph topology
        self.validate_dag()

        # 2. Initialize node statuses
        for node_id in self.nodes:
            context.set_node_status(node_id, "PENDING")

        completed_nodes: Set[str] = set()
        failed_nodes: Set[str] = set()
        skipped_nodes: Set[str] = set()
        in_progress: Set[str] = set()

        processed_nodes: Set[str] = set()

        while len(processed_nodes) < len(self.nodes):
            # Find nodes ready for execution or skipping
            ready_to_run: List[DAGNode] = []
            newly_skipped: List[str] = []

            for node_id, node in self.nodes.items():
                if node_id in processed_nodes or node_id in in_progress:
                    continue

                deps = set(node.depends_on)
                satisfied_deps = deps.intersection(completed_nodes | skipped_nodes)

                # Check if any parent dependency failed
                failed_parents = deps.intersection(failed_nodes)
                if failed_parents:
                    # Parent failed -> skip this node
                    context.set_node_status(
                        node_id,
                        status="SKIPPED",
                        skip_reason=f"Dependency node(s) failed: {', '.join(failed_parents)}",
                    )
                    skipped_nodes.add(node_id)
                    processed_nodes.add(node_id)
                    continue

                # If all dependencies satisfied (completed or skipped)
                if len(satisfied_deps) == len(deps):
                    # Evaluate condition predicate if provided
                    if node.condition is not None:
                        try:
                            should_run = node.condition(context)
                        except Exception as cond_err:
                            logger.error(f"Error evaluating condition for node {node_id}: {cond_err}")
                            should_run = False

                        if not should_run:
                            context.set_node_status(
                                node_id,
                                status="SKIPPED",
                                skip_reason="Condition predicate evaluated to False.",
                            )
                            skipped_nodes.add(node_id)
                            processed_nodes.add(node_id)
                            continue

                    ready_to_run.append(node)

            if not ready_to_run and len(processed_nodes) < len(self.nodes):
                # Safety check against deadlock
                unprocessed = set(self.nodes.keys()) - processed_nodes - in_progress
                for node_id in unprocessed:
                    context.set_node_status(
                        node_id,
                        status="SKIPPED",
                        skip_reason="Unresolvable dependency bottleneck.",
                    )
                    skipped_nodes.add(node_id)
                    processed_nodes.add(node_id)
                break

            if not ready_to_run:
                break

            # Mark ready nodes as in-progress
            for node in ready_to_run:
                in_progress.add(node.node_id)
                context.set_node_status(node.node_id, "RUNNING")

            # Execute ready nodes concurrently using asyncio.gather
            tasks = [
                self._run_node(node, context) for node in ready_to_run
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process execution results
            for node, result in zip(ready_to_run, results):
                node_id = node.node_id
                in_progress.remove(node_id)
                processed_nodes.add(node_id)

                if isinstance(result, Exception):
                    failed_nodes.add(node_id)
                else:
                    completed_nodes.add(node_id)

        context.end_time = time.perf_counter()
        return context

    async def _run_node(self, node: DAGNode, context: PipelineContext) -> Any:
        """Executes a single node with retry logic, timing, and fallback handling."""
        node_id = node.node_id
        start = time.perf_counter()
        attempts = 0
        max_attempts = max(1, node.retry_count + 1)

        last_exception: Optional[Exception] = None

        while attempts < max_attempts:
            attempts += 1
            try:
                res = node.handler(context)
                if inspect.isawaitable(res):
                    res = await res

                duration = (time.perf_counter() - start) * 1000
                context.set_node_status(node_id, status="COMPLETED", duration_ms=duration)
                if res is not None:
                    context.set_output(node_id, res)
                return res
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"Node '{node_id}' failed attempt {attempts}/{max_attempts}: {exc}"
                )
                if attempts < max_attempts:
                    await asyncio.sleep(0.05 * attempts)

        duration = (time.perf_counter() - start) * 1000

        # Attempt fallback if registered
        if node.fallback:
            try:
                logger.info(f"Invoking fallback handler for node '{node_id}'")
                fallback_res = node.fallback(context, last_exception)
                if inspect.isawaitable(fallback_res):
                    fallback_res = await fallback_res

                context.set_node_status(
                    node_id,
                    status="COMPLETED",
                    duration_ms=duration,
                    error_message=f"Primary failed: {last_exception}. Used fallback.",
                )
                if fallback_res is not None:
                    context.set_output(node_id, fallback_res)
                return fallback_res
            except Exception as fb_exc:
                logger.error(f"Fallback for node '{node_id}' also failed: {fb_exc}")
                last_exception = fb_exc

        err_msg = str(last_exception)
        context.add_error(f"Node '{node_id}' failed: {err_msg}")
        context.set_node_status(
            node_id, status="FAILED", duration_ms=duration, error_message=err_msg
        )
        raise last_exception
