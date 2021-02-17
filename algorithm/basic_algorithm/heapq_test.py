import heapq

li = [9, 5, 7, 8, 2, 6, 4, 1, 3]
# heapq.heapify(li)
# print(li)
# heapq.heappush(li, 0)
# print(li)
# item = heapq.heappop(li)
# print(item)
# print(li)

print(heapq.nlargest(5, li))
