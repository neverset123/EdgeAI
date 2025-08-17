## Edge AI
Serving YOLOv8 in browser using tensorflow.js
with `webgl` backend.

### YOLO
https://neverset123.github.io/EdgeAI/
#### Model

   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="tfjs")
   ```
#### Class
src/utils/labels.json

#### Publish
```
git subtree push --prefix dist origin gh-pages
# or
npm run deploy
```