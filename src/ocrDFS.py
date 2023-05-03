import pytesseract
from PIL import Image
import numpy as np
import queue

# Load the image


# Define the connectivity of the pixels
connectivity = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Define the heuristic function
def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


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
        if (img_array[x][y] == 0).all:
            text += pytesseract.image_to_string(Image.fromarray(img_array))
        for dx, dy in connectivity:
            nx, ny = x + dx, y + dy
            if np.logical_and(0 <= nx < img_array.shape[0] , 0 <= ny < img_array.shape[1]):  # Fix the issue here
                stack.append((nx, ny))
        break
    return text


def dfs_main(imgfile):
    img = Image.open(imgfile)
    img_array = np.array(img)
    start = (0, 0)
    goal = (img_array.shape[0] - 1, img_array.shape[1] - 1)
    text_dfs = dfs(img_array, start)

    # Print the recognized text

    return text_dfs