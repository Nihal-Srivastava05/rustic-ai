from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

from jsonata import Jsonata
from pydantic import BaseModel

from rustic_ai.core.agents.commons.message_formats import ErrorMessage
from rustic_ai.core.guild import agent
from rustic_ai.core.guild.agent import Agent, ProcessContext
from rustic_ai.core.guild.dsl import BaseAgentProps
from rustic_ai.core.utils.json_utils import JsonDict

# Format selectors


class PayloadWithFormat(BaseModel):
    payload: JsonDict
    format: str


class BaseFormatSelector(BaseModel):
    strategy: str

    @abstractmethod
    def get_formats(self, elements: List[JsonDict]) -> List[PayloadWithFormat]:
        pass


class FixedFormatSelector(BaseFormatSelector):
    strategy: Literal["fixed"] = "fixed"
    fixed_format: str

    def get_formats(self, elements) -> List[PayloadWithFormat]:
        result = []
        for item in elements:
            result.append(PayloadWithFormat(payload=item, format=self.fixed_format))
        return result


class ListFormatSelector(BaseFormatSelector):
    strategy: Literal["list"] = "list"
    format_list: List[str]

    def get_formats(self, elements) -> List[PayloadWithFormat]:
        result = []
        for item, fmt in zip(elements, self.format_list):
            result.append(PayloadWithFormat(payload=item, format=fmt))
        return result


class DictFormatSelector(BaseFormatSelector):
    strategy: Literal["dict"] = "dict"
    format_dict: dict

    def get_formats(self, elements) -> List[PayloadWithFormat]:
        formats = [value for key, value in sorted(self.format_dict.items())]
        result = []
        for item, fmt in zip(elements, formats):
            result.append(PayloadWithFormat(payload=item, format=fmt))
        return result


class JsonataFormatSelector(BaseFormatSelector):
    strategy: Literal["jsonata"] = "jsonata"
    jsonata_expr: str

    def get_formats(self, elements) -> List[PayloadWithFormat]:
        expr = Jsonata(self.jsonata_expr)
        payload_with_format = [PayloadWithFormat(payload=item, format=expr.evaluate(item)) for item in elements]
        return payload_with_format


# Splitter


class BaseSplitter(BaseModel, ABC):
    split_type: str

    @abstractmethod
    def split(self, payload: Union[str, JsonDict]) -> List[JsonDict]:
        pass


class ListSplitter(BaseSplitter):
    split_type: Literal["list"] = "list"
    field_name: str

    def split(self, payload: JsonDict) -> List[JsonDict]:
        if isinstance(payload[self.field_name], list):
            return payload[self.field_name]
        return [payload[self.field_name]]


class DictSplitter(BaseSplitter):
    split_type: Literal["dict"] = "dict"
    field_name: Optional[str] = None

    def split(self, payload: JsonDict) -> List[JsonDict]:
        if self.field_name:
            [value for key, value in sorted(payload[self.field_name].items())]

        return [value for key, value in sorted(payload.items())]


class JsonataSplitter(BaseSplitter):
    split_type: Literal["jsonata"] = "jsonata"
    expression: str

    def split(self, payload: JsonDict) -> List[JsonDict]:
        evaluator = Jsonata(self.expression)
        result = evaluator.evaluate(payload)
        if isinstance(result, list):
            return result
        elif result is not None:
            return [result]
        return []


class SplitterConf(BaseAgentProps):
    splitter: Union[ListSplitter, JsonataSplitter, DictSplitter]
    format_selector: Union[FixedFormatSelector, ListFormatSelector, DictFormatSelector, JsonataFormatSelector]


class SplitterAgent(Agent[SplitterConf]):
    def __init__(self):
        self.splitter = self.config.splitter
        self.format_selector = self.config.format_selector

    @agent.processor(JsonDict)
    def split_and_send(self, ctx: ProcessContext[JsonDict]) -> None:
        try:
            items = self.splitter.split(ctx.payload)
            payload_with_format = self.format_selector.get_formats(items)

            if len(payload_with_format) != len(items):
                ctx.send_error(
                    ErrorMessage(
                        agent_type=self.get_qualified_class_name(),
                        error_type="LengthMismatch",
                        error_message=f"Number of formats: {len(payload_with_format)} is not same as number of items {len(items)}",
                    )
                )
                return

            for res in payload_with_format:
                ctx.send_dict(payload=res.payload, format=res.format)
        except Exception as e:
            ctx.send_error(
                ErrorMessage(
                    agent_type=self.get_qualified_class_name(),
                    error_type="SplitterError",
                    error_message=str(e),
                )
            )
