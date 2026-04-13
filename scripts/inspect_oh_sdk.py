import inspect
import sys

targets = [
    ("openhands.sdk", "Agent"),
    ("openhands.sdk", "Tool"),
    ("openhands.sdk.tool", "ToolDefinition"),
    ("openhands.sdk.tool", "ToolExecutor"),
    ("openhands.sdk.agent.base", "AgentBase"),
    ("openhands.sdk", "Conversation"),
    ("openhands.sdk.conversation.impl.local_conversation", "LocalConversation"),
]

out_path = sys.argv[1] if len(sys.argv) > 1 else "oh_sdk_inspect.txt"

with open(out_path, "w") as f:
    for module_name, class_name in targets:
        f.write(f"{'='*60}\n{module_name}.{class_name}\n{'='*60}\n")
        try:
            import importlib
            module = importlib.import_module(module_name)
            obj = getattr(module, class_name)
            f.write(inspect.getsource(obj))
        except Exception as e:
            f.write(f"ERROR: {e}\n")
        f.write("\n\n")

print(f"Written to {out_path}")
