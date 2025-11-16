import os
import time
import cv2
import openvino as ov
from dotenv import load_dotenv
from ultralytics import YOLO

# from template_model import TemplateModel # to run in machine
from machine.template_model import TemplateModel  # to run in main

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEBUG = False

class RoboflowModel(TemplateModel):
    def __init__(self):
        self.model = get_model(model_id="nhrl-robots/14", api_key=ROBOFLOW_API_KEY) # TODO

    def predict(self, img, confidence_threshold=0.5, show=False, track=False):
        out = self.model.infer(img)

        bots = {}
        bots["housebot"] = []
        bots["bots"] = []

        preds = out[0].predictions

        housebot_candidates = []

        for pred in preds:
            if pred.confidence > confidence_threshold:
                p = {}
                if pred.class_id == 0:
                    name = "housebot"
                    housebot_candidates.append(pred)
                elif pred.class_id == 1:
                    name = "bots"

                x, y, box_width, box_height = pred.x, pred.y, pred.width, pred.height
                if DEBUG:
                    print(
                        f"x: {x}, y: {y}, box_width: {box_width}, box_height: {box_height}"
                    )

                p["center"] = [x, y]

                img_height, img_width = img.shape[:2]
                x_min = max(0, int(x - box_width / 2) - 20)
                y_min = max(0, int(y - box_height / 2) - 20)
                x_max = min(img_width, int(x + box_width / 2) + 20)
                y_max = min(img_height, int(y + box_height / 2) + 20)

                # Extract bounding box
                screenshot = img[y_min:y_max, x_min:x_max]

                if DEBUG:
                    cv2.imshow("Screenshot", screenshot)
                    cv2.waitKey(0)

                p["bbox"] = [[x_min, y_min], [x_max, y_max]]
                p["img"] = screenshot

                if name == "bots":
                    bots[name].append(p)

        # Select the most confident housebot
        if housebot_candidates:
            best_housebot = max(housebot_candidates, key=lambda pred: pred.confidence)

            # Process the most confident housebot
            x, y, box_width, box_height = (
                best_housebot.x,
                best_housebot.y,
                best_housebot.width,
                best_housebot.height,
            )
            p = {
                "center": [x, y],
                "bbox": [
                    [int(x - box_width / 2) - 20, int(y - box_height / 2) - 20],
                    [int(x + box_width / 2) + 20, int(y + box_height / 2) + 20],
                ],
                "img": img[
                    int(y - box_height / 2) - 20 : int(y + box_height / 2) + 20,
                    int(x - box_width / 2) - 20 : int(x + box_width / 2) + 20,
                ],
            }
            bots["housebot"].append(p)

        if show:
            self.show_predictions(img, bots)
        return bots

    def show_predictions(self, img, predictions):
        # Display housebot
        housebots = predictions["housebot"]
        bots = predictions["bots"]

        color = (0, 0, 255)  # Red for housebots
        for housebot in housebots:
            x_min, y_min = housebot["bbox"][0]
            x_max, y_max = housebot["bbox"][1]
            center_x, center_y = housebot["center"]

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(img, "housebot", (int(x_min), int(y_min - 10)), FONT, 0.5, color, 2)

        color = (255, 255, 255)  # White for bots
        for bot in bots:
            x_min, y_min = bot["bbox"][0]
            x_max, y_max = bot["bbox"][1]
            center_x, center_y = bot["center"]

            # Draw the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label text
            cv2.putText(img, "bot", (int(x_min), int(y_min - 10)), FONT, 0.5, color, 2)

        # print(f"Detected [{len(housebots)} housebots], [{len(bots)} bots]")

        # for name, data in predictions.items():
        #     # Extract bounding box coordinates and class details
        #     x_min, y_min = data['bbox'][0]
        #     x_max, y_max = data['bbox'][1]
        #     center_x, center_y = data['center']

        #     # Choose color based on the class
        #     if 'housebot' in name:
        #         color = (0, 0, 255)  # Red for housebot
        #     else:
        #         color = (0, 255, 0)  # Green for bots

        #     # Draw the bounding box
        #     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        #     # Add label text
        #     cv2.putText(
        #         img, name,
        #         (int(x_min), int(y_min - 10)),  # Slightly above the top-left corner
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5, color, 2
        #     )

        # Display the image with predictions
        # cv2.imshow("Roboflow Predictions", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def train(self, batch, epoch, train_path, validation_path, save_path, save):
        return super().train(batch, epoch, train_path, validation_path, save_path, save)

    def evaluate(self, test_path):
        return super().evaluate(test_path)


class YoloModel(TemplateModel):
    # General template for using YOLO to load model files and use them.

    def __init__(self, model_name, model_type, device=None):
        match model_type:
            case "TensorRT":
                # Works best on NVIDIA GPUs, engine file must be compiled on the PC that it is running on.
                model_extension = ".engine"
                self.model = YOLO("./machine/models/" + model_name + model_extension)
            case "ONNX":
                # Optimal for CPU performance
                model_extension = ".onnx"
                self.model = YOLO("./machine/models/" + model_name + model_extension)
            case "PT":
                # Default kinda
                model_extension = ".pt"
                self.model = YOLO("./machine/models/" + model_name + model_extension)
            case "OpenVIVO":
                # Optimal for Intel CPUs, needs a lil work
                model_extension = ".xml"
                weights_extension = ".bin"
                core = ov.Core()
                classification_model_xml = ("./machine/models/" + model_name + model_extension)
                weights = "./machine/models/" + model_name + weights_extension

                model = core.read_model(model=classification_model_xml, weights=weights)
                cmodel = core.compile_model(model=model)
                self.model = cmodel
        self.device = device
        # compiled_model = core.compile_model(model=model, device_name=device.value)

    def predict(self, img, show=False, track=False):
        # This prints timing info
        if self.device != None:
            results = self.model(img, device=self.device, verbose=False)
        else:
            results = self.model(img)
        # If multiple img passed, results has more than one element
        result = results[0]

        robots = []
        housebots = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy, width, height = box.xywh[0].tolist()
            cropped_img = img[int(y1) : int(y2), int(x1) : int(x2)]

            # cv2.imshow('image', cropped_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows

            dict = {"bbox": [[max(0, x1), max(0, y1)], [min(700, x2), min(700, y2)]], "center": [cx, cy], "img": cropped_img}

            if box.cls == 0:
                housebots.append(dict)
            else:
                robots.append(dict)

        out = {"bots": robots, "housebot": housebots}
        if show:
            self.show_predictions(img, out)

        return out

    def show_predictions(self, img, bots_dict):
        for label, bots in bots_dict.items():

            for bot in bots:

                # Extract bounding box coordinates and class details
                x_min, y_min = bot["bbox"][0]
                x_max, y_max = bot["bbox"][1]

                # Choose color based on the class
                if "housebot" in label:
                    color = (0, 0, 255)  # Red for housebot
                else:
                    color = (255, 255, 255)  # White for bots

                # Draw the bounding box
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

                # Add label text
                cv2.putText(img, label, (int(x_min), int(y_min - 10)), FONT, 0.5, color, 2)

        # cv2.imshow("YoloModel Predictions", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img


# Main code block
if __name__ == "__main__":

    print("starting testing with PT model")
    # predictor = YoloModel("100epoch11","PT")
    predictor = RoboflowModel()

    img_path = (
        os.getcwd() + "/main_files/12567_png.rf.6bb2ea773419cd7ef9c75502af6fe808.jpg"
    )
    img = cv2.imread(img_path)

    # cv2.imshow("Original image", img)
    # cv2.waitKey(0)

    start_time = time.time()
    bots = predictor.predict(img, show=True)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"elapsed time: {elapsed:.4f}")

    # predictor.show_predictions(img, bots)
