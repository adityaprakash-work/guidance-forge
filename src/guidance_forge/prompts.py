# ---INFO-------------------------------------------------------------------------------
"""
Prompts.
"""

# ---DEPENDENCIES-----------------------------------------------------------------------

# --------------------------------------------------------------------------------------
tool_usage_system_prompt = """\
List of GPA TOOLS:

{toolset_info}
"""

tool_determination_prompt = """\
Assess whether an external tool is required at this point in the conversation. If so, 
formulate an execution request detailing the task, including relevant parameters, 
numeric values, and expected outcomes.

Important: If no tool is required then generate a special keyword - NO_TOOL_NEEDED.
Please don't write any other text in the output in this case.

Important: 
- Explain what you want in words not in json.
- parameter name, function name etc. should not be guessed until explicitly mentioned.
- If parameter values are mentioned, they should be used as is and necessarily.
"""

grammarless_polymorphic_agent_system_prompt = """\
You oversee AI agents, including one that calls tools. Identify when a tool is needed.
Signal the tool agent to act accordingly if required.
"""
