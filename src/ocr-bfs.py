import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the image
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding to convert the image to black and white
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Apply BFS to segment the image into characters
def bfs(image, x, y, visited, character):
    if visited[x, y]:
        return
    visited[x, y] = True
    character.append((x, y))
    neighbors = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    for nx, ny in neighbors:
        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and not visited[nx, ny] and image[nx, ny] == 255:
            bfs(image, nx, ny, visited, character)

visited = np.zeros_like(thresh, dtype=bool)
characters = []
for x in range(thresh.shape[0]):
    for y in range(thresh.shape[1]):
        if not visited[x, y] and thresh[x, y] == 255:
            character = []
            bfs(thresh, x, y, visited, character)
            if character:
                characters.append(character)

# Load the training data
data = np.loadtxt("training_data.txt", delimiter=",")
X_train, y_train = data[:, :-1], data[:, -1]

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Classify each character using KNN
result = ""
for character in characters:
    x_min = min(x for x, y in character)
    x_max = max(x for x, y in character)
    y_min = min(y for x, y in character)
    y_max = max(y for x, y in character)
    roi = thresh[x_min:x_max+1, y_min:y_max+1]
    resized_roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    flattened_roi = resized_roi.flatten()
    prediction = knn.predict([flattened_roi])
    result += chr(int(prediction))

print(result)
