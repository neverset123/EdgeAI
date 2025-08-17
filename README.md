## Edge AI
Serving YOLOv8 in browser using tensorflow.js
with `webgl` backend.

### YOLO
#### Model

   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="tfjs")
   ```
#### Class
src/utils/labels.json
