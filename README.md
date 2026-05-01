# Autonomous 25-26

Autonomous perception, orientation, decision, and motor-control code for Combat Robotics at Cornell's robot Huey.

Huey NHRL page: <https://www.nhrl.io/wiki/index.php/Huey>

<img src="docs/images/HueyImage.png" alt="Huey Robot Image" width="50%" />

## What This Repo Runs

The main runtime is [main.py](main.py). It takes a camera feed or a saved video, warps the arena into a top-down coordinate system, detects robots with a YOLO model, identifies Huey by color, estimates Huey's heading from colored corner markers and IMU data, computes drive commands with the RamRam algorithm, and optionally sends those commands over serial to the transmitter hardware.

The actual runtime loop is split into two threads:

- A perception/control thread reads frames, warps them, runs object detection, quantizes Huey's colors, estimates corners/orientation, runs the algorithm, and sends motor commands.
- The main thread handles OpenCV windows and keyboard input.

The processing order is:

1. Capture a frame from `CameraStream` or `cv2.VideoCapture`.
2. Capture one setup frame by pressing `0`.
3. Load or create arena calibration data from `main_files/homography_matrix.txt`.
4. Load or create Huey color selections from `main_files/selected_colors.txt`.
5. Precompute warp maps with `warp_main.py`.
6. Load a YOLO model through `machine/predict.py`.
7. Detect robots in the warped arena image.
8. Quantize detected robot crops with `color_quant/quantization.py`.
9. Identify Huey and estimate its orientation with `corner_detection/`.
10. Optionally fuse/replace orientation with IMU yaw from `sensors/imu_class.py`.
11. Compute `speed` and `turn` with `algorithm/ram.py`.
12. Optionally transmit commands through `transmission/`.
13. Display overlays and write runtime logs.

## Important Files

- `main.py` - full integration entry point.
- `main_helpers.py` - setup helpers, model backend selection, color quantization wrapper, display overlays.
- `camera_stream.py` - threaded low-latency camera capture. Defaults to 1280x720 at 120 FPS.
- `warp_main.py` - homography selection, warp-map generation, and frame warping.
- `machine/predict.py` - YOLO model wrapper used by `main.py`.
- `machine/models/` - local model artifacts.
- `corner_detection/` - Huey color identification and corner/orientation estimation.
- `color_quant/` - LAB-space color snapping for robot crops.
- `algorithm/` - RamRam movement logic.
- `transmission/` - serial and motor-control support for the FlySky/Arduino path.
- `sensors/` - ESP/IMU reading code.
- `runtimesheet/` - timing spreadsheet and plot generation.
- `main_files/` - calibration files and test videos used by `main.py`.
- `quant_settings.json` - tuned color-quantization thresholds and weights.

## Runtime Modes

Set exactly one `MODE` near the top of `main.py`.

### `MODE = "video"`

Best first run. Uses a saved video, disables transmission, disables the camera capture thread, and processes at `FRAME_RATE = 60`.

You must also set:

```python
camera_number = folder + "/test_videos/huey_vs_prince.mp4"
camera_type = "Video"
```

### `MODE = "live"`

Uses a live camera and the threaded `CameraStream`. In the current code, `live` also sets `IS_TRANSMITTING = True`, so only use it when the serial transmitter path is available or change `IS_TRANSMITTING` manually after the mode block.

### `MODE = "comp"`

Competition mode. Uses live camera input, enables transmission, and turns the weapon path on by default.

### `MODE = "custom"`

Leaves the mode-specific overrides alone so you can manually set camera, transmission, display, and timing options.

## Setup Guide

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd Autonomous-25-26
```

### 2. Use the expected Python version

`requirements.txt` currently documents Python `3.13.12`.

Check your version:

```bash
python3 --version
```

On Windows, use `python --version`.

### 3. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main runtime depends on OpenCV, NumPy, pandas, PyTorch, Ultralytics, OpenVINO, PySerial, OpenPyXL, Plotly, and Core ML tooling.

### 5. Confirm model files exist

`main.py` currently defaults to:

```python
MODEL_NAME = "Nano320Temp"
OD_IMG_SIZE = 320
```

That means the model loader looks under:

```text
machine/models/Nano320Temp/320/
```

The repo currently includes `Nano320Temp.pt` and `Nano320Temp.onnx` there. Other model families are also present, including `NanoSizeVariant`, `NanoDefault`, `NanoSegHueyPrince`, and `SmallComp`.

Backend selection happens in `main_helpers.get_predictor()`:

- CUDA available: TensorRT `.engine`
- Apple MPS available: CoreML `.mlpackage`
- OpenVINO GPU available: OpenVINO model directory
- OpenVINO CPU available: OpenVINO model directory
- Otherwise: ONNX CPU

Make sure the artifact for your selected backend actually exists. For example, `Nano320Temp/320` currently has `.pt` and `.onnx`, but not every backend artifact.

### 6. Check camera access

To list cameras with names:

```bash
python see_all_cameras.py
```

To quickly probe indices `0` through `4`:

```bash
python check_number_of_cameras.py
```

Set `camera_number` in `main.py` to the index you want. `CameraStream` uses AVFoundation on macOS, the default backend on Windows/Linux, MJPG, 1280x720, 120 FPS, and a capture buffer size of 1.

### 7. Decide whether you need hardware

For saved-video testing, you do not need robot hardware, a transmitter, or an IMU.

For a live camera-only test, use `MODE = "custom"` or manually disable:

```python
IS_TRANSMITTING = False
IMU_ENABLED = False
```

For full robot operation, you need:

- A camera or capture card.
- Arduino/FlySky transmission hardware. See [transmission/README.md](transmission/README.md).
- ESP/IMU hardware if `IMU_ENABLED = True`. See [sensors/README.md](sensors/README.md).

If `IMU_ENABLED = True`, startup will ask you to choose the ESP serial port when no port is passed. If `IS_TRANSMITTING = True`, startup will ask you to choose the Arduino serial port when needed.

## First Recommended Run

Start with a saved video and no hardware. In `main.py`, set:

```python
MODE = "video"
IMU_ENABLED = False
WARP_AND_COLOR_PICKING = False
SHOW_FRAME = True
SHOW_HUD = True
SHOW_QUANTIZED_HUEY = True
camera_number = folder + "/test_videos/huey_vs_prince.mp4"
camera_type = "Video"
```

Then run:

```bash
python main.py
```

When the first window appears, press `0` to capture the setup frame. With `WARP_AND_COLOR_PICKING = False`, the program then loads:

```text
main_files/homography_matrix.txt
main_files/selected_colors.txt
```

The first ML/corner-detection result is shown before the match loop starts. Press a key in that window to continue.

## Calibration

Calibration has two parts:

- Arena homography: maps camera coordinates into a 700x700 arena image.
- Huey colors: stores the colors used for Huey body/front/back corner detection.

### Reuse saved calibration

This is the default:

```python
WARP_AND_COLOR_PICKING = False
```

The runtime reads:

```text
main_files/homography_matrix.txt
main_files/selected_colors.txt
```

### Create new calibration

Set:

```python
WARP_AND_COLOR_PICKING = True
```

Then run `main.py`.

1. Press `0` to capture the setup frame.
2. Select arena corners in order: top left, top right, bottom right, bottom left.
3. Press `z` while selecting corners to undo the previous point.
4. Pick Huey's relevant colors when the color picker appears.
5. The new calibration files are written into `main_files/`.

Use `DISPLAY_SCALE` if the setup windows are too large or too small.

## Quantization Settings

`quant_settings.json` contains named color-quantization presets. `main.py` currently loads:

```python
quantization_settings = all_settings["Green Huey"]
```

Available presets currently include:

- `Ryan OG Green Settings`
- `Green Huey High-T`
- `Purple Huey`
- `Green Huey`

If Huey is identified inconsistently after object detection works, this file and `main_files/selected_colors.txt` are the first places to check.

## Keyboard Controls

During the OpenCV runtime:

- `q` - quit.
- `f` - toggle manual flipped-drive direction.
- `p` - pause or resume.
- `w` - toggle weapon state in shared state.
- Any other key while paused - step one frame.
- `r` - passed to the RamRam algorithm to reset recovery history.

During the initial setup-frame window:

- `0` - capture the current frame.
- `q` - quit without capturing.

During arena corner selection:

- `z` - undo the previous selected corner.
- `Esc` - leave selection.

## Runtime Outputs

When `SHEET_RUNTIME = True`, `RuntimeSheet` collects per-frame timing data and saves files on cleanup:

```text
runtimesheet/itertimes.xlsx
runtimesheet/itertimes.png
runtimesheet/itertimes.svg
runtimesheet/itertimes_stacked.png
runtimesheet/itertimes_stacked.svg
runtimesheet/itertimes_interactive.html
runtimesheet/itertimes_interactive_stacked.html
```

Corner-detection color percentage rows are saved to:

```text
color_output.csv
```

## Common Problems

### The program asks for a serial port when I only want to test video

Set `IMU_ENABLED = False` and make sure `IS_TRANSMITTING = False`. `MODE = "live"` currently turns transmission on automatically.

### The camera opens but FPS or resolution is wrong

Check the printed capture properties from `CameraStream`. OpenCV camera property requests are not guaranteed to be honored by every camera/backend.

### No robots are detected

Check:

- The selected `MODEL_NAME` and `OD_IMG_SIZE`.
- Whether the matching model artifact exists for the backend selected on your machine.
- Whether the input is already warped as expected.
- Lighting, exposure, and arena visibility.
- The YOLO thresholds in `machine/predict.py`.

### Huey is detected as the enemy, or orientation is unstable

Check:

- `main_files/selected_colors.txt`
- `quant_settings.json`
- `SHOW_QUANTIZED_HUEY`
- Whether the YOLO crop cuts off Huey's colored markers.
- Whether `IMU_ENABLED` is overriding camera orientation during low-corner cases.

### Runtime is too slow

Check `runtimesheet/itertimes.xlsx` or the generated plots. Common quick changes:

- Use a smaller/faster model.
- Reduce `OD_IMG_SIZE`.
- Disable `SHOW_QUANTIZED_HUEY`.
- Disable `SHEET_RUNTIME` during competition.
- Disable `SHOW_HUD` or `DISPLAY_ANGLES` if display drawing is a bottleneck.
- Avoid per-frame debug printing.

## Related Docs

- [algorithm/README.md](algorithm/README.md)
- [corner_detection/README.md](corner_detection/README.md)
- [transmission/README.md](transmission/README.md)
- [sensors/README.md](sensors/README.md)
- [vid_and_img_processing/README.md](vid_and_img_processing/README.md)
