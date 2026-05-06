"""Interactive CLI for testing the agent loop in isolation.

Usage (from project root):
    python -m server.cli

Commands:
    quit / exit       end the session
    reset             clear conversation history (keeps the same session id)
    state             print the current trip state
"""

import json
import uuid

from server import state as state_module
from server.llm import MODEL, run_agent


def main():
    session_id = str(uuid.uuid4())
    history: list[dict] = []

    print(f"Lake District Trip Planner — model: {MODEL}")
    print(f"session: {session_id[:8]}")
    print("Type 'quit' to exit, 'reset' to clear history, 'state' to inspect trip state.")
    print()

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            break
        if user.lower() == "reset":
            history = []
            state_module.reset_session(session_id)
            print("[history and trip state cleared]\n")
            continue
        if user.lower() == "state":
            print(json.dumps(state_module.get_trip_state(session_id), indent=2))
            print()
            continue

        try:
            result = run_agent(session_id, user, history)
        except Exception as e:
            print(f"[ERROR] {e.__class__.__name__}: {e}\n")
            continue

        history = result["history"]
        if result["tool_calls_made"]:
            print(f"[tools: {', '.join(result['tool_calls_made'])}]")
        print(f"\nBot: {result['reply']}\n")


if __name__ == "__main__":
    main()
