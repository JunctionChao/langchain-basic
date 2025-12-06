import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver


load_dotenv()

DB_URI = os.getenv("DB_URI")
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpoints = checkpointer.list(
    #     config={"configurable": {"thread_id": "1"}}
    # )
    # for checkpoint in checkpoints:
    #     print(checkpoint)
    #     messages = checkpoint[1]["channel_values"]["messages"]
    #     for message in messages:
    #         message.pretty_print()
    #     break

    checkpoint = checkpointer.get({"configurable": {"thread_id": "1"}})
    print(checkpoint)
    messages = checkpoint["channel_values"]["messages"]
    for message in messages:
        message.pretty_print()
    


    # messages = checkpointer.get(thread)

