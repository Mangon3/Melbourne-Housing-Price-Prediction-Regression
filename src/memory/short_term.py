from typing import List, Dict
class ShortTermMemory:
    """
    Simples in-memory store for the last N conversation turns.
    Used for providing context to the Agent's planning module.
    """

    def __init__(self, limit: int = 10):
        self.history: List[Dict[str, str]] = []
        self.limit = limit

    def add_turn(self, user_input: str, agent_response: str):
        """Adds a turn and maintains the limit."""
        if not user_input or not agent_response:
            return
        self.history.append({"role": "user", "content": user_input})
        clean_response = agent_response[:200] + "..." if len(agent_response) > 200 else agent_response
        self.history.append({"role": "agent", "content": clean_response})
        if len(self.history) > self.limit * 2:
            self.history = self.history[-(self.limit * 2):]

    def get_context_string(self) -> str:
        """Returns the history formatted for the LLM prompt."""
        if not self.history:
            return "No previous context."
        return "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in self.history])

    def clear(self):
        self.history = []
        
stm = ShortTermMemory(limit=3)  
