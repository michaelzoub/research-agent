"""Explicit tool layer for model-directed agents."""

from .base import BaseTool, ToolContext, ToolRegistry, ToolResult
from .code import CodeExecutionTool
from .files import FileReadTool
from .graders import CompareCandidateToChampionTool, PredictionMarketEvaluationTool, evaluator_context
from .research import DocumentFigureTool, SearchTool, WebFetchTool
from .document_data import DocumentAnalysisTool, SVGChartTool, StructuredDataExtractionTool
from .terminal import TerminalExecutionTool
from .learnings import SaveLearningTool

__all__ = ["BaseTool", "CodeExecutionTool", "CompareCandidateToChampionTool", "DocumentAnalysisTool", "DocumentFigureTool", "FileReadTool", "PredictionMarketEvaluationTool", "SaveLearningTool", "SearchTool", "SVGChartTool", "StructuredDataExtractionTool", "TerminalExecutionTool", "ToolContext", "ToolRegistry", "ToolResult", "WebFetchTool", "evaluator_context"]
