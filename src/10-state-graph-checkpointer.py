# 通过 checkepointer 保存 checkpoint （每一步的图状态快照）
# graph 状态图执行后，可以通过 thread_id （checkpoint的身份识别） 访问图状态
# 通过上述原理可以实现 
# - memory 记忆管理
# - time travel 时间旅行/情景重现
# - human-in-the-loop 暂停/人为交互
# - fault-tolerance 容错/回退重试


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add


class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

# 全局状态图
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# checkpointer存储每一个步骤状态图的快照checkpoint
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 通过配置thread_id 可以区分不同会话, 识别同一会话的checkpoint
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

res = graph.invoke({"foo": "", "bar":[]}, config)
print(res) # {'foo': 'b', 'bar': ['a', 'b']}

# 获取thred_id最新的状态快照checkpoint
latest_checkpoint = graph.get_state(config)
print("最后一个快照:\n", latest_checkpoint)
# StateSnapshot(
#   values={'foo': 'b', 'bar': ['a', 'b']}, 
#   next=(), 
#   config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d278f-7298-6f90-8002-b2ae84636144'}}, 
#   metadata={'source': 'loop', 'step': 2, 'parents': {}}, 
#   created_at='2025-12-06T07:55:43.308993+00:00', 
#   parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d278f-7296-6882-8001-abb17e90cdde'}}, 
#   tasks=(), 
#   interrupts=()
# )

# 获取倒数第二个状态快照checkpoint的checkpoint_id
checkpoint_id_2ndlast = latest_checkpoint.parent_config["configurable"]["checkpoint_id"]
# 可以通过 checkpoint_id 访问特定的状态快照
config_with_cpid = {"configurable": {"thread_id": "1", "checkpoint_id": checkpoint_id_2ndlast}}
checkpoint_2ndlast = graph.get_state(config_with_cpid)
print("倒数第二个快照:\n", checkpoint_2ndlast)


# 列出所有状态快照checkpoint
for checkpoint_tuple in checkpointer.list(config):
    print()
    print(checkpoint_tuple[2]["step"], checkpoint_tuple[2]["source"])
    print(checkpoint_tuple[1]["channel_values"])

    # print(checkpoint_tuple)
'''
CheckpointTuple(
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d2a9d-3108-695c-bfff-d6ec784ef82f'}}, 
    checkpoint={'v': 4, 'ts': '2025-12-06T13:45:28.086767+00:00', 
        'id': '1f0d2a9d-3108-695c-bfff-d6ec784ef82f', 
        'channel_versions': {'__start__': '00000000000000000000000000000001.0.053918949303447206'}, 
        'versions_seen': {'__input__': {}}, 
        'updated_channels': ['__start__'], 
        'channel_values': {'__start__': {'foo': '', 'bar': []}}
    }, 
    metadata={'source': 'input', 'step': -1, 'parents': {}}, 
    parent_config=None, 
    pending_writes=[
        ('94e1e481-636c-de6a-1acf-b75d2d01e428', 'foo', ''), 
        ('94e1e481-636c-de6a-1acf-b75d2d01e428', 'bar', []), 
        ('94e1e481-636c-de6a-1acf-b75d2d01e428', 'branch:to:node_a', None)
    ]
)

CheckpointTuple(
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d2a9d-310b-6067-8000-0c64233ea152'}}, 
    checkpoint={'v': 4, 'ts': '2025-12-06T13:45:28.087767+00:00', 
        'id': '1f0d2a9d-310b-6067-8000-0c64233ea152', 
        'channel_versions': {'__start__': '00000000000000000000000000000002.0.28175628555423826', 'foo': '00000000000000000000000000000002.0.28175628555423826', 'bar': '00000000000000000000000000000002.0.28175628555423826', 'branch:to:node_a': '00000000000000000000000000000002.0.28175628555423826'}, 
        'versions_seen': {
            '__input__': {}, 
            '__start__': {'__start__': '00000000000000000000000000000001.0.053918949303447206'}
        }, 
        'updated_channels': ['bar', 'branch:to:node_a', 'foo'], 
        'channel_values': {'foo': '', 'bar': [], 'branch:to:node_a': None}}, 
    metadata={'source': 'loop', 'step': 0, 'parents': {}}, 
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0d2a9d-3108-695c-bfff-d6ec784ef82f'}}, 
    pending_writes=[
        ('8bfa14fa-faeb-d273-b3a6-4cf00c6f84ff', 'foo', 'a'), 
        ('8bfa14fa-faeb-d273-b3a6-4cf00c6f84ff', 'bar', ['a']), 
        ('8bfa14fa-faeb-d273-b3a6-4cf00c6f84ff', 'branch:to:node_b', None)
    ]
)

'''


def save_graph_visualization(graph, filename: str = "graph.png") -> None:  
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
    except IOError as e:
        print(f"Failed to save graph visualization: {e}")

# save_graph_visualization(graph, "imgs/graph.png")