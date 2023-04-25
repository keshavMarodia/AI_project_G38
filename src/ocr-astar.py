import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from collections import deque
from queue import PriorityQueue

def astar(img, start, end):
    # Define the heuristic function
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    # Define the cost function
    def cost(current, next):
        return np.sqrt((next[0] - current[0])**2 + (next[1] - current[1])**2)
    
    # Define the neighbors function
    def neighbors(point):
        x, y = point
        return [(x-1, y-1), (x-1, y), (x-1, y+1),
                (x, y-1),             (x, y+1),
                (x+1, y-1), (x+1, y), (x+1, y+1)]
    
    # Initialize the start node and the end node
    start_node = (start, 0, heuristic(start, end))
    end_node = (end, float('inf'), 0)
    
    # Initialize the open and closed sets
    open_set = PriorityQueue()
    open_set.put(start_node)
    closed_set = set()
    
    # Initialize the dictionary to store the path and its cost
    path = {}
    path[start] = None
    
    # Loop until the open set is empty
    while not open_set.empty():
        # Get the node with the lowest f-score
        current_node = open_set.get()
        current = current_node[0]
        
        # Check if the current node is the end node
        if current == end_node[0]:
            # Reconstruct the path and return it
            path_points = []
            while current is not None:
                path_points.append(current)
                current = path[current]
            return path_points[::-1]
        
        # Add the current node to the closed set
        closed_set.add(current)
        
        # Loop through the neighbors of the current node
        for neighbor in neighbors(current):
            # Check if the neighbor is inside the image boundaries and is not in the closed set
            if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < img.shape[1] and neighbor[1] < img.shape[0] and neighbor not in closed_set:
                # Calculate the tentative g-score for the neighbor
                tentative_g_score = current_node[1] + cost(current, neighbor)
                
                # Check if the neighbor is already in the open set
                neighbor_node = None
                for n in list(open_set.queue):
                    if n[0] == neighbor:
                        neighbor_node = n
                        break
                
                # If the neighbor is not in the open set, add it with the tentative g-score and f-score
                if neighbor_node is None:
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    open_set.put((neighbor, tentative_g_score, f_score))
                    path[neighbor] = current
                # If the neighbor is already in the open set, update its g-score and f-score if the tentative g-score is lower
                elif tentative_g_score < neighbor_node[1]:
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    open_set.put((neighbor, tentative_g_score, f_score))
                    path[neighbor] = current
    
    # If the end node is not reached, return an empty path
    return []

digits = datasets.load_digits()

train_data = digits.data
train_data = np.array(train_data, dtype=np.float32)
print(train_data)
train_labels = digits.target

# Load the image
img = cv2.imread(r'C:\medAi\htr tf2\SimpleHTR\data\ocrdemo.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocess the image
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours from left to right
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# Load the KNN model
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Classify each component using the trained KNN model
result = ""
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = thresh[y:y+h, x:x+w]
    # marodia pitega
    
    # Resize the component to 28x28 pixels
    resized_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Flatten the image to a 1D numpy array
    flattened_roi = resized_roi.reshape((1, 28*28))
    
    # Normalize the image
    normalized_roi = flattened_roi / 255.0
    
    # Use the KNN model to predict the digit
    print(knn.findNearest(normalized_roi, k=5))
    ret, result_, neighbours, dist = knn.findNearest(normalized_roi, k=5)
    result += str(int(result_[0][0]))
    
    # Find the center of the component
    center_x = x + w/2
    center_y = y + h/2
    
    # Find the center of the image
    image_center_x = img.shape[1]/2
    image_center_y = img.shape[0]/2
    
    # Calculate the distance between the component center and the image center
    distance = np.sqrt((center_x - image_center_x)**2 + (center_y - image_center_y)**2)
    
    # Use A* algorithm to find the shortest path to the component
    path = astar(thresh, (int(image_center_x), int(image_center_y)), (int(center_x), int(center_y)))
    
    # Draw the path on the image
    for p in path:
        cv2.circle(img, (p[0], p[1]), 1, (0, 255, 0), thickness=-1)
    
    # Draw a circle at the center of the component
    cv2.circle(img, (int(center_x), int(center_y)), int(distance), (0, 0, 255), thickness=2)

# Print the result
print(result)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
