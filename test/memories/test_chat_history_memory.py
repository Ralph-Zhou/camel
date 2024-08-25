# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import tempfile
from pathlib import Path

import pytest

from camel.memories import ChatHistoryMemory, MemoryRecord
from camel.memories.context_creators import ScoreBasedContextCreator
from camel.messages import BaseMessage, Content
from camel.storages.key_value_storages import (
    InMemoryKeyValueStorage,
    JsonStorage,
)
from camel.types import ModelType, OpenAIBackendRole, RoleType
from camel.utils.token_counting import OpenAITokenCounter


@pytest.fixture
def memory(request):
    context_creator = ScoreBasedContextCreator(
        OpenAITokenCounter(ModelType.GPT_4), ModelType.GPT_4.token_limit
    )
    if request.param == "in-memory":
        yield ChatHistoryMemory(
            context_creator=context_creator, storage=InMemoryKeyValueStorage()
        )
    elif request.param == "json":
        _, path = tempfile.mkstemp()
        path = Path(path)
        yield ChatHistoryMemory(
            context_creator=context_creator, storage=JsonStorage(path)
        )
        path.unlink()


@pytest.mark.parametrize("memory", ["in-memory", "json"], indirect=True)
def test_chat_history_memory(memory: ChatHistoryMemory):
    system_msg = BaseMessage(
        role_name="system",
        role_type=RoleType.DEFAULT,
        meta_dict=None,
        content=Content(text="You are a helpful assistant"),
    )
    user_msg = BaseMessage(
        role_name="AI user",
        role_type=RoleType.USER,
        meta_dict=None,
        content=Content(text="Do a task"),
    )
    assistant_msg = BaseMessage(
        role_name="AI assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content=Content(text="OK"),
    )
    system_record = MemoryRecord(system_msg, OpenAIBackendRole.SYSTEM)
    user_record = MemoryRecord(user_msg, OpenAIBackendRole.USER)
    assistant_record = MemoryRecord(assistant_msg, OpenAIBackendRole.ASSISTANT)
    memory.write_records([system_record, user_record, assistant_record])
    output_messages, _ = memory.get_context()
    assert output_messages[0] == system_msg.to_openai_system_message()
    assert output_messages[1] == user_msg.to_openai_user_message()
    assert output_messages[2] == assistant_msg.to_openai_assistant_message()
