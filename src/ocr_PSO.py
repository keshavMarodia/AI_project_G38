import pytesseract
from PIL import Image
import numpy as np
import queue

# Load the image
img = Image.open('line4.jpeg')

# Convert the image to a numpy array
img_array = np.array(img)

# Define the connectivity of the pixels
connectivity = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Define the heuristic function
def heuristic(node, goal):
    return abs(node - goal) + abs(node[1] - goal[1])

# Define the A* search function

# Define the BFS function
def bfs(img_array, start):
    q = queue.Queue()
    q.put(start)
    visited = set()
    text = ''
    while not q.empty():
        node = q.get()
        if node in visited:
            continue
        visited.add(node)
        x, y = node
        if img_array[x][y] == 0:
            text += pytesseract.image_to_string(Image.fromarray(img_array))
        for dx, dy in connectivity:
            nx, ny = x + dx, y + dy
            if 0 <= nx < img_array.shape[0] and 0 <= ny < img_array.shape[1]:
                q.put((nx, ny))
    return text

# Define the DFS function
def dfs(img_array, start):
    stack = [start]
    visited = set()
    text = ''
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        x, y = node
        if img_array[x][y] == 0:
            text += pytesseract.image_to_string(Image.fromarray(img_array))
        for dx, dy in connectivity:
            nx, ny = x + dx, y + dy
            if 0 <= nx < img_array.shape[0] and 0 <= ny < img_array.shape[1]:
                stack.append((nx, ny))
    return text

# Perform A* search, BFS, and DFS on the image
start = (0, 0)
goal = (img_array.shape[0] - 1, img_array.shape[1] - 1)  # Fix the issue here
text_astar = astar(img_array, start, goal)
text_bfs = bfs(img_array, start)
text_dfs = dfs(img_array, start)

# Print the recognized text
print('A*:', text_astar)
print('BFS:', text_bfs)
print('DFS:', text_dfs)