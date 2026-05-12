## Edge AI

https://neverset123.github.io/EdgeAI/

- Serving YOLOv8 in browser using tensorflow.js with `webgl` backend.
- Serving light-weighted guppylm-9M chat model

```
npm install
npm run build
npm run deploy # push dist dir to branch gh-pages
```

## Others
1. YOLO model in python
class definition: src/utils/labels.json
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="tfjs")
```
2. BYOM
https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb
