"""Processing stages for data transformation."""

from hermes.stages.data_loader_stage import DataLoaderStage
from hermes.stages.llm_invocation_stage import LLMInvocationStage
from hermes.stages.multi_run_stage import (
    AggregationStrategy,
    AllStrategy,
    AverageStrategy,
    ConsensusStrategy,
    FirstSuccessStrategy,
    MultiRunStage,
)
from hermes.stages.parser_factory import create_response_parser
from hermes.stages.pipeline_stage import PipelineStage
from hermes.stages.prompt_formatter_stage import (
    PromptFormatterStage,
)
from hermes.stages.response_parser_stage import (
    JSONParser,
    PydanticParser,
    RawTextParser,
    RegexParser,
    ResponseParser,
    ResponseParserStage,
)
from hermes.stages.result_writer_stage import ResultWriterStage
from hermes.stages.stage_registry import StageRegistry, stage

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
