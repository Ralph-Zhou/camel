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
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import torch
from openai import Stream
from outlines import generate, models
from pydantic import BaseModel

from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ModelType,
)
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
)

T = TypeVar('T', bound=BaseModel)


class SchemaModel(BaseModelBackend):
    r"""Shema model in a unified BaseModelBackend interface, which aims to
    generate the formatted response."""

    def __init__(
        self,
        model_type: ModelType,
        model_config_dict: Dict[str, Any],
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        r"""Constructor for OpenAI backend.

        Args:
            model_type (ModelType): Model for which a backend is created,
                one of GPT_* series.
            model_config_dict (Dict[str, Any]): A dictionary that will
                be fed into openai.ChatCompletion.create().
            api_key (Optional[str]): The API key for authenticating with the
                OpenAI service. (default: :obj:`None`)
            url (Optional[str]): The url to the OpenAI service.
        """
        super().__init__(model_type, model_config_dict, api_key, url)
        self._url = url
        self._api_key = api_key
        self._client = Union[models.Transformers, models.LlamaCpp, models.VLLM]
        self.device: str = "cuda"
        match self.model_type:  # match the requested model type
            case ModelType.TRANSFORMERS:
                model_name = self.model_config_dict.get(
                    "model_name", "mistralai/Mistral-7B-v0.3"
                )
                self.device = self.model_config_dict.get("device", "cuda")
                self._client = models.transformers(
                    model_name=model_name,
                    device=self.device,
                )
            case ModelType.LLAMACPP:
                self.device = self.model_config_dict.get("device", "cuda")
                repo_id = self.model_config_dict.get(
                    "repo_id", "TheBloke/phi-2-GGUF"
                )
                filename = self.model_config_dict.get(
                    "filename", "phi-2.Q4_K_M.gguf"
                )
                self._client = models.llamacpp(
                    repo_id=repo_id, filename=filename
                )
            case ModelType.VLLM:
                self.device = self.model_config_dict.get("device", "cuda")
                model_name = self.model_config_dict.get(
                    "model_name", "mistralai/Mistral-7B-v0.3"
                )
                self._client = models.vllm(model_name=model_name)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        self._token_counter: Optional[BaseTokenCounter] = None

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(self.model_type)
        return self._token_counter

    def run(
        self,
        messages: List[OpenAIMessage],
        class_schema: Optional[Type[T]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of schema-based chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            pydantic_class (Optional[Type[T]]): Pydantic class schema to
                generate the response. (default: :obj:`None`)

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk], T]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        if class_schema is None:  # class_schema is required
            raise ValueError(
                "Pydantic class schema is required for schema-based model."
            )

        generator = generate.json(self._client, class_schema)

        # Set seed for reproducibility
        rng = torch.Generator(device=self.device)
        rng.manual_seed(789001)

        messages_str = "\n".join(
            [
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
            ]
        )
        parsed_response = generator(messages_str, rng=rng)

        print(repr(parsed_response))  # TODO: Remove this line

        import time

        response = ChatCompletion(
            id=f"chatcmpl-{time.time()}",
            created=int(time.time()),
            model=str(self.model_type),
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content=str(parsed_response)
                    ),
                    finish_reason="stop",
                )
            ],
        )

        return response

    def check_model_config(self):
        r"""Check whether the model configuration contains the required
        arguments for the schema-based model.

        Raises:
            Warning: If the model configuration dictionary does not contain
                the required arguments for the schema-based model, the warnings
                are raised.
        """
        # Check the model_name, WarningError if not found
        if "model_name" not in self.model_config_dict:
            raise Warning("The model_name is set to the default value.")
        if "device" not in self.model_config_dict:
            raise Warning("The device is set to the default value cuda.")

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode,
            which sends partial results each time.
        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)