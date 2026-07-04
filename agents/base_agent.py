# ============================================================
#  agents/base_agent.py
#  HermesAgent — Base class that mirrors the Hermes Agent
#  design pattern (NousResearch/hermes-agent).
#
#  Hermes Agent is a system-level CLI agent framework, not a
#  pip-installable library. This base class implements the same
#  conceptual architecture:
#    - Each agent has a clear ROLE (identity/purpose statement)
#    - Each agent has TOOLS (callable functions it can invoke)
#    - Each agent implements a STEP method (one unit of work)
#    - The FEEDBACK LOOP agent uses run_loop() for the agent-loop
#      pattern, running iteratively and updating internal state
#      between iterations — matching Hermes Agent's loop/feedback
#      design philosophy.
#
#  Reference: https://github.com/NousResearch/hermes-agent
# ============================================================

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class AgentResult:
    """
    Structured output passed between agents in the pipeline.
    Every agent's step() method returns an AgentResult.
    """
    agent_name: str
    success: bool
    data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ OK" if self.success else "❌ FAILED"
        return f"[{self.agent_name}] {status} | data_keys={list(self.data.keys())}"


class HermesAgent(ABC):
    """
    Abstract base class implementing the Hermes Agent design pattern.

    Each concrete agent must define:
      - name:  Human-readable identifier
      - role:  System prompt / role description (used in LLM calls)
      - tools: List of callable tool functions this agent can use
      - step(**kwargs): One unit of work; returns AgentResult

    The run_loop() method enables the feedback-loop pattern used by
    Agent 5 (FeedbackLoopAgent).
    """

    def __init__(
        self,
        name: str,
        role: str,
        tools: Optional[list[Callable]] = None,
    ) -> None:
        self.name = name
        self.role = role
        self.tools: list[Callable] = tools or []
        self._log = get_logger(f"agent.{name.lower().replace(' ', '_')}")

        self._log.info(
            f"[bold]Agent initialized:[/bold] [cyan]{self.name}[/cyan] | "
            f"tools={[t.__name__ for t in self.tools]}"
        )

    @abstractmethod
    def step(self, **kwargs) -> AgentResult:
        """
        Execute one step of this agent's task.
        Must be implemented by every concrete agent.

        Returns:
            AgentResult with success flag, data payload, and any warnings/errors.
        """
        ...

    def run_loop(
        self,
        iterations: int = 5,
        delay_seconds: int = 300,
        **step_kwargs,
    ) -> list[AgentResult]:
        """
        Agent feedback-loop pattern (Hermes Agent design).

        Runs step() repeatedly, waits delay_seconds between iterations,
        and accumulates results. The loop updates internal state between
        iterations so each cycle can learn from the previous one.

        Args:
            iterations:     Number of loop cycles to run.
            delay_seconds:  Wait time between iterations (default: 300s = 5 min).
            **step_kwargs:  Passed to step() on each iteration.

        Returns:
            List of AgentResult, one per iteration.
        """
        results = []
        self._log.info(
            f"[bold]Starting feedback loop:[/bold] [cyan]{self.name}[/cyan] | "
            f"iterations={iterations}, delay={delay_seconds}s"
        )

        for i in range(iterations):
            self._log.info(
                f"[cyan]{self.name}[/cyan] — loop iteration {i+1}/{iterations}"
            )
            result = self.step(iteration=i, **step_kwargs)
            results.append(result)

            if i < iterations - 1:
                self._log.info(
                    f"[cyan]{self.name}[/cyan] — waiting {delay_seconds}s before next cycle…"
                )
                time.sleep(delay_seconds)

        self._log.info(
            f"[cyan]{self.name}[/cyan] — loop complete. "
            f"Successes: {sum(1 for r in results if r.success)}/{iterations}"
        )
        return results
