from ultralytics import YOLO

models_folder = "./machine/models/"

# PRE-REQ: Make a new folder in the model name folder with your desired image size as the name, and drag the pt file into there.

# model_name = "SmallComp"
# model_name = "NanoSizeVariant"
model_name = "Nano320Temp"
# model_name = "NanoSegHueyPrince"

# Smaller number -> Faster
desired_model_input_size = 320

# CoreML for M-series macs, engine for NVIDIA gpus
# desired_format = "coreml"
# desired_format = "engine"
desired_format = "onnx"
# desired_format = "openvino"

base_model_extension = ".pt"

# Load the YOLO model
model = YOLO(models_folder + model_name + "/" + str(desired_model_input_size) +
             "/" + model_name + base_model_extension, task='detect')

print(model.export(format=desired_format, imgsz=desired_model_input_size,
      half=True, simplify=True, task='detect'))

# Terminal prompt: yolo export model=./machine/models/SmallComp/416/SmallComp.pt format=engine simplify=True imgsz=416 half=True
