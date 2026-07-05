# Unit tests for the rule-based memory classifier policy.
from app.memory_classifier import classify_memory


# Find the first classified memory of a given type.
def has_type(memories: list[dict], memory_type: str) -> dict:
    for item in memories:
        if item.get("memory_type") == memory_type:
            return item
    raise AssertionError(f"Missing memory_type={memory_type}. Memories={memories}")


# Run memory classification policy checks.
def main() -> None:
    # Available food should be short-lived working memory.
    item = has_type(classify_memory("Yes I have bread in the kitchen"), "working_memory")
    assert item["ttl_minutes"] == 30
    assert item["confidence"] > 0

    # Current symptoms should be temporary working memory.
    item = has_type(classify_memory("I feel dizzy after standing up"), "working_memory")
    assert item["ttl_minutes"] == 120

    # Scam or medication incidents should become episodic memory.
    item = has_type(classify_memory("Someone emailed me a link asking for money"), "episodic_memory")
    assert item["ttl_minutes"] is None

    # Confirmed caregiver facts should update long-term profile.
    item = has_type(classify_memory("My daughter is Sara and she can help me"), "long_term_profile")
    assert item["confirmed"] is True

    # Confirmed health conditions should be stored under health profile metadata.
    item = has_type(classify_memory("I have diabetes"), "long_term_profile")
    assert item["metadata"]["profile_category"] == "health_condition"

    # General knowledge questions should not become confirmed profile facts.
    item = has_type(classify_memory("Is banana good for empty stomach?"), "semantic_knowledge")
    assert item["confirmed"] is False

    # LLM error turns should never become user memory.
    item = has_type(
        classify_memory(
            "I feel dizzy",
            final_message_source="conversation_agent_error",
            llm_error={"error_type": "ProviderError"},
        ),
        "do_not_save",
    )
    assert item["confidence"] == 1.0

    print("All memory classifier tests passed.")


# Allow direct command-line execution.
if __name__ == "__main__":
    main()
