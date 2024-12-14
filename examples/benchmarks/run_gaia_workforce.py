# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========


from camel.agents import ChatAgent
from camel.benchmarks import DefaultGAIARetriever, GAIABenchmark
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.runtime import RemoteHttpRuntime
from camel.societies.workforce import Workforce
from camel.toolkits import *
from camel.toolkits.base import BaseToolkit
from camel.types import ModelPlatformType, ModelType, StorageType

from typing import List, Dict
from dotenv import load_dotenv

import os
import json

load_dotenv()

# TODO: Implement the missing tools
missing_required_tools = [
    "Computer Vision Tools",
    "Graph Interaction Tools",
    "Bass Note Data",
    "C++ Compiler",
    "Text Editor",
    "Text Processing/Diff Tool",
    "YouTube",
    "Video Parsing",
    "Access to Academic Journal Websites",
    "Audio Processing Software",
    "YouTube Player",
    "Speech-to-Text Audio Processing Tool",
    "Google Translate Access",
    "GIF Parsing Tools",
    "Access to Google Maps",
    "Babylonian Cuneiform to Arabic Legend",
    "Image Processing Tools",
    "Counter",
    "Code/Data Analysis Tools",
    "Spreadsheet Editor",
    "Video Processing Software",
    "Audio Capability",
    "Color Recognition",
    "Image Recognition Tools (including OCR)",
    "Word Document Access",
    "Rubik's Cube Model",
    "Computer Algebra System",
    "PowerPoint Viewer"
]


def _process_tools(tools: List[str] | str) -> List[FunctionTool]:
    r"""Process the tools from the configuration."""
    
    tool_list = []
    if isinstance(tools, str):
        tools = [tools]
    for tool_name in tools:
        if tool_name in globals():
            toolkit_class: BaseToolkit = globals()[tool_name]
            tool_list.extend(toolkit_class().get_tools())
        else:
            raise ValueError(f"Toolkit {tool_name} not found.")


def load_config(agent_system_config_path: str) -> Workforce:
    r"""Load the agent system configuration from a JSON file, and return the workforce."""
    
    # Load the agent system configuration
    with open(agent_system_config_path, "r") as file:
        agent_system_config: List[str] = json.load(file)
    file.close()
    
    default_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )
    workforce = Workforce("GAIA Workforce")
    
    for agent_config in agent_system_config:
        
        sys_msg = BaseMessage.make_assistant_message(
            role_name=agent_config["role_name"],
            content=agent_config["sys_msg_content"]
        )
        
        _tools = _process_tools(agent_config["tools"])
        
        agent = ChatAgent(
            system_message=sys_msg,
            model=default_model,
            tools=_tools
        )
        
        workforce.add_single_agent_worker(
            description=agent_config["description"],
            worker=agent
        )
    
    print("Agent system configuration loaded.")
    return workforce


def test_config(agent_system_config_path: str):

    retriever = DefaultGAIARetriever(
        vector_storage_local_path="local_data2/", storage_type=StorageType.QDRANT
    )

    benchmark = GAIABenchmark(
        data_dir="datasets_test",
        processes=1,
        save_to="results.jsonl",
        retriever=retriever,
    )

    print(f"Number of validation examples: {len(benchmark.valid)}")
    print(f"Number of test examples: {len(benchmark.test)}")


    toolkit = CodeExecutionToolkit(verbose=True)
    runtime = RemoteHttpRuntime("localhost").add(
        toolkit.get_tools(),
        "camel.toolkits.CodeExecutionToolkit",
    )

    workforce = load_config(agent_system_config_path)

    result = benchmark.run_workforce(workforce, "valid", level=1, subset=4)
    print("correct:", result["correct"])
    print("total:", result["total"])



if __name__ == "__main__":
    
    agent_system_config_path = "agent_system_config.json"
    
    test_config(agent_system_config_path)