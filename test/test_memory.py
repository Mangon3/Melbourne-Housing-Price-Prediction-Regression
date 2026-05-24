from src.memory.short_term import ShortTermMemory

def test_short_term_memory_add():
    stm = ShortTermMemory(limit=2)
    stm.add_turn("user1", "agent1")
    stm.add_turn("user2", "agent2")
    assert len(stm.history) == 4
    
    # Exceed limit
    stm.add_turn("user3", "agent3")
    assert len(stm.history) == 4
    assert stm.history[0]["content"] == "user2"
    
def test_short_term_memory_empty():
    stm = ShortTermMemory()
    stm.add_turn("", "")
    assert len(stm.history) == 0

def test_get_context_string():
    stm = ShortTermMemory(limit=2)
    assert stm.get_context_string() == "No previous context."
    stm.add_turn("Hello", "World")
    ctx = stm.get_context_string()
    assert "USER: Hello" in ctx
    assert "AGENT: World" in ctx

def test_clear():
    stm = ShortTermMemory(limit=2)
    stm.add_turn("A", "B")
    stm.clear()
    assert len(stm.history) == 0
