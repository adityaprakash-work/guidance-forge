# ---INFO-------------------------------------------------------------------------------
"""
Temporary API for building agentic systems.
"""

# ---DEPENDENCIES-----------------------------------------------------------------------
from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from textwrap import dedent
from typing import TYPE_CHECKING

import guidance
from guidance import assistant, gen, user

from .prompts import (
    grammarless_polymorphic_agent_system_prompt,
    tool_determination_prompt,
    tool_usage_system_prompt,
)

if TYPE_CHECKING:
    from guidance._grammar import RawFunction
    from guidance._guidance import GuidanceFunction
    from guidance.models._model import Model


# --------------------------------------------------------------------------------------
def trace_fn(fn, *args, **kwargs):
    buffer = StringIO()
    with redirect_stdout(buffer):
        ret = fn(*args, **kwargs)
    out = buffer.getvalue()
    return out, ret


def tool(fn: Callable) -> GuidanceFunction:
    info = dedent(f"""\
    ------------
    GPA_TOOL
    Name       :    {fn.__name__}
    Description:    {inspect.getdoc(fn)}
    Annotations:    {inspect.get_annotations(fn)}\n
    """)

    # NOTE: Don't use role blocks within the wrapper. The grammar is meant to be used
    # within an external assistant block.
    @guidance(dedent=False)
    def wrapper(lm: Model, *args, **kwargs):
        out, ret = trace_fn(fn, *args, **kwargs)
        lm += f"stdout: {out}\n"
        lm += f"return: {ret}\n\n"
        return lm

    wrapper.name = fn.__name__
    wrapper.info = info
    return wrapper


@guidance
def invoke(lm: Model, user_prompt: str):
    with user():
        lm += user_prompt
    return lm


class GrammarlessPolymorphicAgent:
    def __init__(
        self,
        lm: Model,
        system_prompt: str = None,
        tools: Iterable[Callable] = None,
    ):
        self._lm = lm
        self.system_prompt = (
            system_prompt
            if system_prompt
            else grammarless_polymorphic_agent_system_prompt
        )
        self._tools = {tool.name: tool for tool in tools} if tools else {}
        self.toolset_info = (
            "\n".join(tool.info for tool in tools)
            if tools
            else "No tools are available."
        )
        with self.toggle_ev(self._lm):
            # NOTE: Can't send system blocks to some newer models. There is a separate
            # parameter for this in the model constructor. No covered by guidance.
            with user():
                self._lm_chat = self._lm + self.system_prompt
                self._lm_tool_unused = (
                    self._lm
                    + tool_usage_system_prompt.format(toolset_info=self.toolset_info)
                    + dedent("""\
                This is a proxy system block for you. Adhere to it but do not
                mention it further in the conversation.
                """)
                )
        self._lm_chat.echo = True
        self._lm_tool = self._lm_tool_unused.copy()
        self._lm_tool_counter = 0

    def reset_tool_agent(self):
        self._lm_tool_counter += 1
        self._lm_tool_counter %= 3
        if self._lm_tool_counter == 0:
            self._lm_tool = self._lm_tool_unused.copy()

    def __add__(self, invocation: RawFunction) -> GrammarlessPolymorphicAgent:
        self._lm_chat += invocation
        self._lm_chat += self.determine_tool_usage()
        if "NO_TOOL_NEEDED" not in (tud := self._lm_chat["tool_usage_directive"]):
            self._lm_tool += self.set_tool_invocations(tud)
            tool_invocations = json.loads(self._lm_tool["tool_invocations"])
            with assistant():
                self._lm_chat += "Invoking Tools...\n\n"
                for ti in tool_invocations:
                    name = ti["name"]
                    prms = ti["parameters"]
                    meta = ti["meta"]
                    tool = self._tools[name]
                    self._lm_chat += f"> {name} {prms} {meta}\n"
                    self._lm_chat += tool(**prms)
                self._lm_chat += dedent("""\
                    Now, getting back to the conversation with these results ...
                    """)
        with assistant():
            self._lm_chat += gen()
        self.reset_tool_agent()

        return self

    @staticmethod
    @guidance
    def determine_tool_usage(lm: Model):
        temp = lm.copy()
        with user():
            temp += tool_determination_prompt
        with assistant():
            temp += gen(name="assessment")
        lm = lm.set("tool_usage_directive", temp["assessment"])
        return lm

    @guidance
    def set_tool_invocations(self, lm: Model, tool_usage_directive: str):
        with user():
            lm += tool_usage_directive
            lm += dedent("""\
            You will now list down names of the tools under GPA TOOLS that have to be 
            used to solve the task at hand. Output this as a comma separated list 
            without any other text.
            """)
        with assistant():
            lm += gen(name="list_tools")
        tool_invocations = []
        for tool_name in lm["list_tools"].split(","):
            tool_name = tool_name.strip()
            if tool_name in self._tools:
                with user():
                    temp = lm + dedent(f"""\
                    Now you will give parameters for {tool_name} in a dictionary format 
                    so that it can be called given the task at hand. You will not write 
                    anything other than the dictionary.
                    """)
                with assistant():
                    temp += gen(name="parameters")
                with user():
                    temp += dedent("""\
                    Why did You invoke this tool in brief:
                    """)
                with assistant():
                    temp += gen(name="meta")
                self.haha = temp
                # Cut out section between { and } in temp["parameters"] and replace
                # single quotes with double quotes.
                # TODO: single quotes with double quotes is a hack. Something better?
                prms = temp["parameters"]
                prms = prms[prms.find("{") : prms.find("}") + 1]
                prms = prms.replace("'", '"')
                prms = json.loads(prms)
                invo = {
                    "name": tool_name,
                    "parameters": prms,
                    "meta": temp["meta"],
                }
                tool_invocations.append(invo)
        lm = lm.set("tool_invocations", json.dumps(tool_invocations))
        return lm

    @staticmethod
    @contextmanager
    def toggle_ev(lm: Model):
        o = lm.echo
        lm.echo = not o
        try:
            yield
        finally:
            lm.echo = o
