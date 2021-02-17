# https://zhuanlan.zhihu.com/p/151380132


graph = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D", "E"],
    "D": ["B", "C", "E", "F"],
    "E": ["D", "C"],
    "F": ["D"]
}


def BFS(graph, s):
    queue = []
    queue.append(s)
    seen = set()
    seen.add(s)
    while len(queue) > 0:
        vetex = queue.pop(0)
        nodes = graph[vetex]
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
        print(vetex)


def DFS(graph, s):
    stack = []
    stack.append(s)
    seen = set()
    seen.add(s)
    while len(stack) > 0:
        vetex = stack.pop()
        nodes = graph[vetex]
        for w in nodes:
            if w not in seen:
                stack.append(w)
                seen.add(w)
        print(vetex)


DFS(graph, "A")
