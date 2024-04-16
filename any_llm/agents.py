import subprocess

from models import get_response, models


def system_message_to_init_history(content):
    return [{"role": "system", "content": content}]


class Agent:
    """A simple scaffolded agent that can do bash calls"""

    def __init__(
        self,
        system_message="You are a helpful assistant",
        temperature=1,
        model=models[0],
        print_messages=False,
        has_bash=False,
    ):
        self.history = system_message_to_init_history(system_message)
        self.init_history = list(self.history)
        self.temperature = temperature
        self.model = model
        self.name = f"{model} @ T={temperature}"
        self.print_messages = print_messages
        self.has_bash = has_bash

    def reply(self, msg, *, reset_conversation=True, seed=None):
        """Add the message to the history and let gpt run until no more <bash>command</bash> is in its output"""
        if reset_conversation:
            self.history = list(self.init_history)
        if self.print_messages:
            print(msg)
        self.history.append({"role": "user", "content": msg})
        while True:
            response = get_response(
                model=self.model,
                history=self.history,
                temperature=self.temperature,
                seed=seed,
            )
            if self.print_messages:
                print(response["content"])
            self.history.append(response)
            if self.has_bash and "<bash>" in response["content"]:
                self.execute(response["content"])
            else:
                return response["content"]

    def execute(self, respone_str: str):
        """Execute the command in the response and add the output to the history"""
        cmd = respone_str.split("<bash>")[1].split("</bash>")[0]
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = f"Output: \n{result.stdout}"
        if result.stderr:
            output += f"\nError captured:\n{result.stderr}"
        self.history.append({"role": "user", "content": output})


def get_agents(
    system_messages=["You are a helpful assistant."],
    models=models,
    temperatures=[0, 1],
    agent_kwargs={},
):
    agents = []
    for system_message in system_messages:
        for model in models:
            for t in temperatures:
                agents.append(
                    Agent(
                        system_message=system_message,
                        model=model,
                        temperature=t,
                        **agent_kwargs,
                    )
                )
    return agents


def cli():
    """Chat with an agent - nice for working on the agent"""
    agent = Agent()
    while True:
        msg = input("> ")
        if msg == "q":
            return
        agent.reply(msg)


if __name__ == "__main__":
    cli()
