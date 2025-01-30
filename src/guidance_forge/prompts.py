# ---INFO-------------------------------------------------------------------------------
"""
Prompts.
"""

# ---DEPENDENCIES-----------------------------------------------------------------------

# --------------------------------------------------------------------------------------
tool_usage_system_prompt = """\
You are an AI agent in-charge of the following tools:
{toolset_info}

When presented with a scenario, your task is to generate a string representation of the
parameters
"""
