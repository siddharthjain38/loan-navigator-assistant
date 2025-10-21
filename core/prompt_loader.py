"""
Prompt loader utility for loading prompts from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class PromptLoader:
    """Loads and manages prompts from YAML files."""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}

    def load_prompts(self, agent_name: str) -> Dict[str, Any]:
        """Load all prompts for a specific agent from YAML file."""
        if agent_name in self._cache:
            return self._cache[agent_name]

        yaml_file = self.prompts_dir / f"{agent_name}.yaml"

        if not yaml_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {yaml_file}")

        with open(yaml_file, "r", encoding="utf-8") as file:
            prompts = yaml.safe_load(file)

        self._cache[agent_name] = prompts
        return prompts

    def get_prompt(self, agent_name: str, prompt_name: str) -> str:
        """Get a specific prompt for an agent."""
        prompts = self.load_prompts(agent_name)

        if prompt_name not in prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found for agent '{agent_name}'")

        return prompts[prompt_name]

    def get_list(self, agent_name: str, list_name: str) -> List[str]:
        """Get a list from agent prompts (like low_confidence_indicators)."""
        prompts = self.load_prompts(agent_name)

        if list_name not in prompts:
            raise KeyError(f"List '{list_name}' not found for agent '{agent_name}'")

        return prompts[list_name]

    def format_prompt(self, agent_name: str, prompt_name: str, **kwargs) -> str:
        """Get and format a prompt with variables."""
        template = self.get_prompt(agent_name, prompt_name)
        return template.format(**kwargs)

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()


# Global instance for easy access
prompt_loader = PromptLoader()
