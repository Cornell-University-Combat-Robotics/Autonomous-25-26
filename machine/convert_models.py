from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./machine/models/26sBest80.pt")

# Export the model to TensorRT format
print(model.export(format="engine", half=True))  # creates 'yolo11n.engine'

# Load the exported TensorRT model
# tensorrt_model = YOLO("100epoch11.engine")
