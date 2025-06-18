from loguru import logger

from beyondagent.client.em_client import EMClient
from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.schema.trajectory import Trajectory


class AgentFlow(BaseAgentFlow):

    def __init__(self, enable_historical_experience: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.enable_historical_experience: bool = enable_historical_experience
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.em_client = EMClient()

    def add_historical_experience(self, trajectory: Trajectory):
        if self.enable_historical_experience:
            history_experience = self.em_client.call_context_generator(trajectory=trajectory,
                                                                       retrieve_top_k=1,
                                                                       workspace_id="default")
            if history_experience:
                logger.info(f"history_experience={history_experience}")
                trajectory.steps[-1]["content"] = history_experience + "\n\n" + trajectory.steps[-1]["content"]

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        if self.enable_historical_experience:  # add by jinli 0618
            self.add_historical_experience(trajectory)

        for act_step in range(self.max_steps):
            prompt_text = self.tokenizer.apply_chat_template(trajectory.steps,
                                                             tokenize=False,
                                                             add_generation_prompt=True)
            current_token_len = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])

            if current_token_len > self.max_model_len:
                logger.warning(f"exceed max model len={self.max_model_len}")
                break

            # callback llm server, messages.size=1
            llm_output = self.llm_chat_fn(trajectory.steps)
            assert len(llm_output) == 1
            trajectory.steps.extend(llm_output)

            env_output = env.step(instance_id, llm_output[0])
            state_content: str = env_output["state"]["content"]
            # yunpeng add: 
            # if env_output["state"]["role"]=="tool":
            #     env_output["state"]["role"] = "user"

            if len(self.tokenizer(state_content, return_tensors="pt", padding=False)["input_ids"][
                       0]) > self.max_env_len:
                env_output["state"]["content"] = state_content[:self.max_env_len]

            trajectory.steps.append(env_output["state"])
            trajectory.is_terminated = env_output["is_terminated"]

            # TODO require env
            # trajectory.reward.outcome = env_output["reward"]["outcome"]
            # trajectory.reward.description = env_output["reward"]["description"]
            trajectory.reward.outcome = env_output["reward"]
            trajectory.reward.description = "Outcome 1 = success, 0 = failure."

            if trajectory.is_terminated:
                break

        if trajectory.steps[-1]["role"] == "user":
            trajectory.steps = trajectory.steps[:-1]

        return trajectory
