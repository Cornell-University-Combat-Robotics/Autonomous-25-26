from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./machine/models/26n320_2.pt")

# Export the model to TensorRT format
# print(model.export(format="engine", half=True))  # creates 'yolo11n.engine'

# Export the model to ONNX format and quantize to INT8

print(model.export(format="onnx", simplify=True, imgsz=320))  # creates 'yolo11n.onnx'

# Load the exported TensorRT model
# tensorrt_model = YOLO("100epoch11.engine")

# Terminal prompt: yolo export model=./machine/models/26n320_2.pt format=onnx simplify=True imgsz=320
