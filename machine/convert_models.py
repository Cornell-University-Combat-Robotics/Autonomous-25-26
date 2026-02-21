from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./machine/models/temp26nBest3.pt")

# Export the model to TensorRT format
# print(model.export(format="engine", half=True))  # creates 'yolo11n.engine'

# Export the model to ONNX format and quantize to INT8

print(model.export(format="onnx", simplify=True))  # creates 'yolo11n.onnx'

# Load the exported TensorRT model
# tensorrt_model = YOLO("100epoch11.engine")
