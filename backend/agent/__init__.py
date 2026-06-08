"""
Autonomous research agent.

A THINK -> EXECUTE -> REFLECT loop that, given a task, writes Python code, runs it
in a throwaway Docker sandbox, checks the result, and refines until it has the best
working solution. Unlike the cited-RAG answer path, this agent *verifies* its answer
by actually running it.

    python -m backend.agent "Find the fastest correct way to test primality up to 10^7"

Public API:
    from backend.agent.loop import run_agent, AgentResult
    from backend.agent.code_runner import run_python, docker_available
"""
from backend.agent.loop import run_agent, AgentResult           # noqa: F401
from backend.agent.code_runner import run_python, docker_available, RunResult  # noqa: F401
from backend.agent.memory import TwoTierMemory                  # noqa: F401

__all__ = ["run_agent", "AgentResult", "run_python", "docker_available",
           "RunResult", "TwoTierMemory"]
