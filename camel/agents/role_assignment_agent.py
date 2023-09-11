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
import re
from typing import Any, Dict, List, Optional, Union

from tenacity import retry, stop_after_attempt, wait_exponential

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.prompts import TextPrompt
from camel.typing import ModelType, RoleType


class RoleAssignmentAgent(ChatAgent):
    r"""An agent that generates role names based on the task prompt.
    Attributes:
        role_assignment_prompt (TextPrompt): A prompt for the agent to generate
        role names.

    Args:
        model (ModelType, optional): The type of model to use for the agent.
            (default: :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        model: ModelType = ModelType.GPT_3_5_TURBO,
        model_config: Optional[Any] = None,
    ) -> None:
        system_message = BaseMessage(
            role_name="Role Assigner",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="You assign roles based on tasks.",
        )
        super().__init__(system_message, model, model_config)

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def run(
        self,
        task_prompt: Union[str, TextPrompt],
        num_roles: int = 2,
    ) -> Dict[str, str]:
        r"""Generate role names based on the input task prompt.

        Args:
            task_prompt (Union[str, TextPrompt]): The prompt
                for the task based on which the roles are to be generated.
            num_roles (int, optional): The number of roles to generate.
                (default: :obj:`2`)

        Returns:
            Dict[str, str]: A dictionary mapping role names to their
                descriptions.
        """
        self.reset()

        task_prompt = TextPrompt("===== TASK =====\n" + task_prompt + "\n\n")
        expert_prompt = "===== ANSWER TEMPLATE =====\n" + "\n".join(
            f"Domain expert {i + 1}: <BLANK>\n"
            f"Associated competencies, characteristics, duties "
            f"and workflows: <BLANK>. End." for i in range(num_roles or 0))
        role_assignment_generation_prompt = TextPrompt(
            "You are a role assignment agent, and you're in charge of " +
            "recruiting {num_roles} experts for the following task." +
            "Identify the domain experts you'd recruit and detail " +
            "descriptions, like their associated competencies, " +
            "characteristics, duties and workflows to complete the task.\n " +
            "Your answer MUST adhere to the format of ANSWER TEMPLATE, and " +
            "ONLY answer the BLANKs.\n" + expert_prompt + task_prompt)
        role_assignment_generation = role_assignment_generation_prompt.format(
            num_roles=num_roles, task=task_prompt)

        role_assignment_generation_msg = BaseMessage.make_user_message(
            role_name="Role Assigner", content=role_assignment_generation)

        response = self.step(input_message=role_assignment_generation_msg)

        msg = response.msg  # type: BaseMessage
        terminated = response.terminated

        # Distribute the output completions into role names and descriptions
        role_names = [
            desc.replace("<|", "").replace("|>", "") for desc in re.findall(
                r"Domain expert \d: (.+?)\nAssociated competencies,",
                msg.content,
                re.DOTALL,
            )
        ]
        role_descriptions = [
            desc.replace("<|", "").replace("|>", "") for desc in re.findall(
                r"Associated competencies, characteristics, "
                r"duties and workflows: (.+?) End.", msg.content, re.DOTALL)
        ]

        if len(role_names) != num_roles or len(role_descriptions) != num_roles:
            raise RuntimeError(
                "Got None or insufficient information of roles.")
        if terminated:
            raise RuntimeError("Role assignment failed.")

        role_descriptions_dict = {
            role_name: description
            for role_name, description in zip(role_names, role_descriptions)
        }

        return role_descriptions_dict

    def run_task_assignment():
        # TODO: Implement task assignment
        print("Not implemented yet.")

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def split_tasks(
        self,
        task_prompt: Union[str, TextPrompt],
        num_subtasks: Optional[int] = None,
        role_descriptions_dict: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        r"""Split the task into subtasks based on the input task prompt.

        Args:
            task_prompt (Union[str, TextPrompt]): The prompt
                for the task based on which the roles are to be generated.
            num_subtasks (Optional[int], optional): The number of subtasks to
                generate. (default: :obj:`None`)
            role_descriptions_dict (Optional[Dict[str, str]], optional): The
                role descriptions of each role. (default: :obj:`None`)

        Returns:
            List[str]: The subtasks to complete the whole task.
        """
        role_descriptions_dict = (role_descriptions_dict
                                  or self.role_descriptions_dict)
        role_names = list(role_descriptions_dict.keys())

        task_prompt = TextPrompt("===== TASK =====\n" + task_prompt + "\n\n")
        role_with_description_prompt = \
            "===== ROLES WITH DESCRIPTION =====\n" + "\n".join(
                f"{role_name}:\n{role_descriptions_dict[role_name]}\n"
                for role_name in role_names) + "\n\n"
        answer_prompt = "===== ANSWER TEMPLATE =====\n" + "\n".join(
            f"Content of subtask {i + 1}: <BLANK>"
            for i in range(num_subtasks or 0)) + "\n\n"
        splict_task_prompt = TextPrompt(
            "You are a role assignment agent, and you're in asked with " +
            "dividing the main TASK into {num_subtasks} subtasks. " +
            "In your team consists of {num_roles} domain experts each " +
            "contributing to the {num_subtasks} subtasks.\n" +
            "Your answer MUST adhere to the format of ANSWER TEMPLATE, and " +
            "ONLY answer the content of subtask without roles in BLANKs.\n\n" +
            answer_prompt + task_prompt + role_with_description_prompt)

        if num_subtasks is None:
            subtasks_generation = splict_task_prompt.format(
                num_subtasks=num_subtasks, num_roles=len(role_names))
        else:
            subtasks_generation = splict_task_prompt.format(
                num_subtasks=num_subtasks, num_roles=len(role_names))
        subtasks_generation_msg = BaseMessage.make_user_message(
            role_name="Task Splitter", content=subtasks_generation)

        response = super().step(input_message=subtasks_generation_msg)

        msg = response.msg  # type: BaseMessage
        terminated = response.terminated

        # Distribute the output completions into subtasks
        subtasks = [
            desc.replace("<|", "").replace("|>", "")
            for desc in re.findall(r"Content of subtask \d: (.+?)(?=\n|$)",
                                   msg.content, re.DOTALL)
        ]

        if num_subtasks is not None and len(subtasks) != num_subtasks:
            raise RuntimeError(
                "Got None or insufficient information of subtasks.")
        if terminated:
            raise RuntimeError("Subtask split failed.")

        self.subtasks = subtasks
        return subtasks
