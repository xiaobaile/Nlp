def DFS(graph, start):
    stack = []
    stack.append(start)
    seen = set()
    seen.add(start)
    while len(stack) > 0:
        temp = stack.pop(-1)
        nodes = graph[temp]
        for node in nodes:
            if node not in seen:
                seen.add(node)
                stack.append(node)
        print(node)