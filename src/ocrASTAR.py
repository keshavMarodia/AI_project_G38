import pytesseract
from PIL import Image
import numpy as np
import queue

# Define the connectivity of the pixels
connectivity = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar(img_array, start, goal):
    q = queue.PriorityQueue()
    q.put((0, start))
    visited = set()
    text = ''
    while not q.empty():
        _, node = q.get()
        if node in visited:
            continue
        visited.add(node)
        x, y = node
        if (img_array[x][y]== 0).all:
            text = pytesseract.image_to_string(Image.fromarray(img_array))
        if node == goal:
            break
        for dx, dy in connectivity:
            nx, ny = x + dx, y + dy
            if np.logical_and(0 <= nx < img_array.shape[0] , 0 <= ny < img_array.shape[1]):  # Fix the issue here
                cost = heuristic((nx, ny), goal)
                q.put((cost, (nx, ny)))
        break
    return text


def astar_main(img_file):
    print(img_file)
    img = Image.open(img_file)
    img_array = np.array(img)
    start = (0, 0)
    goal = (img_array.shape[0] - 1, img_array.shape[1] - 1)
    text_astar = astar(img_array, start, goal)

    # print('A*:', text_astar)
    return text_astar
