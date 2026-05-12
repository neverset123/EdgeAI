## Edge AI
- Serving YOLOv8 in browser using tensorflow.js with `webgl` backend.
- Serving light-weighted guppylm-9M chat model

Frontend: https://neverset123.github.io/EdgeAI/

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

