# ---INFO-------------------------------------------------------------------------------
"""
Temporary API for building agentic systems.
"""

# ---DEPENDENCIES-----------------------------------------------------------------------
from contextlib import contextmanager
from textwrap import dedent
from typing import TYPE_CHECKING

import guidance
from guidance import assistant, gen, silent, system, user

from .prompts import tool_usage_system_prompt

if TYPE_CHECKING:
    from guidance._grammar import RawFunction
    from guidance.models._model import Model


# --------------------------------------------------------------------------------------
def tool(fn):
    name = fn.__name__
    info = dedent(f"""\
    Tool Info  -
    Name       :    {name}
    Description:    {fn.__doc__}
    Annotations:    {fn.__annotations__}
    """)

    @guidance
    def wrapper(lm, *args, **kwargs):
        with user():
            lm += ""
        ...

    return {"info": info, "raw_function": wrapper}


@guidance
def invoke(lm: Model, user_prompt: str):
    with user():
        lm += user_prompt
    return lm


class GrammarlessPolymorphicAgent:
    def __init__(self, lm: Model, system_prompt: str = None, tools: dict = None):
        self._lm = lm
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else "You are an extremely helpful assistant."
        )
        self._tools = tools
        with self.toggle_ev(self._lm):
            self._lm_chat = self._lm + self.system_prompt
            self._lm_tool = self._lm + tool_usage_system_prompt.format()

    def __add__(self, invocation: RawFunction) -> "GrammarlessPolymorphicAgent":
        self._lm_chat += invocation

    @guidance
    def determine_tool_usage(self, lm: Model):
        temp = lm.copy()
        with user():
            temp += ""
        ...

    @staticmethod
    @contextmanager
    def toggle_ev(lm: Model):
        o = lm.echo
        lm.echo = not o
        try:
            yield
        finally:
            lm.echo = o
