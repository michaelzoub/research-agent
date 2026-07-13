"""Explicit tool layer for model-directed agents."""

from .base import BaseTool, ToolContext, ToolRegistry, ToolResult
from .code import CodeExecutionTool
from .files import FileReadTool
from .graders import PredictionMarketEvaluationTool, evaluator_context
from .research import DocumentFigureTool, SearchTool, WebFetchTool
from .terminal import TerminalExecutionTool
from .swarm import OptimizationSwarmTool, ParameterSweepTool, SaveLearningTool

__all__ = ["BaseTool", "CodeExecutionTool", "DocumentFigureTool", "FileReadTool", "OptimizationSwarmTool", "ParameterSweepTool", "PredictionMarketEvaluationTool", "SaveLearningTool", "SearchTool", "TerminalExecutionTool", "ToolContext", "ToolRegistry", "ToolResult", "WebFetchTool", "evaluator_context"]
