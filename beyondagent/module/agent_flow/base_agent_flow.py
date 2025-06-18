from typing import Any, Callable

from beyondagent.client.env_client import EnvClient
from beyondagent.schema.trajectory import Trajectory


class BaseAgentFlow(object):

    def __init__(self,
                 llm_chat_fn: Callable,
                 tokenizer: Any,
                 max_steps: int = 10,
                 max_model_len: int = 20480,
                 max_env_len: int = 1024,
                 **kwargs):
        # super.__init__(**kwargs)
        self.llm_chat_fn: Callable = llm_chat_fn
        self.tokenizer = tokenizer
        self.max_steps: int = max_steps
        self.max_model_len: int = max_model_len
        self.max_env_len: int = max_env_len

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        raise NotImplementedError
