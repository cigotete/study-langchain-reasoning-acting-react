import os
from typing import Union, List

from langchain.agents import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

from callbacks import AgentCallbackHandler # Hook into langchain events
callbackAgent = AgentCallbackHandler() # Create callback instance

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        temperature=0,
        model_kwargs={'stop':["\nObservation"]},
        callbacks=[callbackAgent],
    )
    intermediate_steps = []

    # Agent variable is defined using a chain of three functions using the | operator.
    # Anonymous function: { "input": lambda x: x["input"] }
    # takes a dictionary as input and returns a new dictionary with the same content.
    # Its purpose is to provide an "input" key for later stages of the chain.
    # prompt: This refers to the prompt variable with the template with information about available tools.
    # llm: An instance of the ChatOpenAI class for generating text based on the given prompt.
    #
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # The invoke method of the agent is called with a dictionary containing the input question.
    # This call returns an object of type AgentAction or AgentFinish.
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of characters of the following text: DOG?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    # Checks if the agent_step is an AgentAction object. If it is, steps are performed.
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool # Get tool name
        tool_to_use = find_tool_by_name(tools, tool_name) # Find tool by name
        tool_input = agent_step.tool_input # Get tool input
        observation = tool_to_use.func(str(tool_input)) # Run tool and get observation
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation))) # Add observation to intermediate steps

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of characters of the following text: DOG?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)