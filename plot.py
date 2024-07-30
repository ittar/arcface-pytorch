# import matplotlib.pyplot as plt
import ast
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import pandas as pd

IMAGE_SIZE = 20

def getImage(path, zoom=1):
    img = Image.open(path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)  # Resize the image to 10x10 pixels
    return OffsetImage(np.array(img), zoom=zoom)


df = pd.read_csv('df/p2v_tsne.csv')
paths = df.path.values
x = df.x.values
y = df.y.values

fig, ax = plt.subplots(figsize=(20,20))
ax.scatter(x, y) 

for x0, y0, path in zip(x, y, paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)

plt.title('TSNE Visualization of Face Recognition Using ArcFace')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.show()
