"""Processing stages for data transformation."""

from ceres.stages.data_loader_stage import DataLoaderStage
from ceres.stages.llm_invocation_stage import LLMInvocationStage
from ceres.stages.multi_run_stage import (
    AggregationStrategy,
    AllStrategy,
    AverageStrategy,
    ConsensusStrategy,
    FirstSuccessStrategy,
    MultiRunStage,
)
from ceres.stages.parser_factory import create_response_parser
from ceres.stages.pipeline_stage import PipelineStage
from ceres.stages.prompt_formatter_stage import (
    PromptFormatterStage,
)
from ceres.stages.response_parser_stage import (
    JSONParser,
    PydanticParser,
    RawTextParser,
    RegexParser,
    ResponseParser,
    ResponseParserStage,
)
from ceres.stages.result_writer_stage import ResultWriterStage
from ceres.stages.stage_registry import StageRegistry, stage

__all__ = [
    "PipelineStage",
    "DataLoaderStage",
    "PromptFormatterStage",
    "LLMInvocationStage",
    "ResponseParserStage",
    "ResultWriterStage",
    "MultiRunStage",
    "ResponseParser",
    "RawTextParser",
    "JSONParser",
    "PydanticParser",
    "RegexParser",
    "create_response_parser",
    "AggregationStrategy",
    "ConsensusStrategy",
    "FirstSuccessStrategy",
    "AllStrategy",
    "AverageStrategy",
    # Stage Registry
    "StageRegistry",
    "stage",
]
