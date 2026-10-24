# Deep-Live-Cam: Windows Setup Guide

Real-time face swap that you can use in video calls through OBS Virtual Camera.
This guide takes you from a fresh unzip to a working live call.

Time needed: about 30-45 minutes, mostly downloads.

---

## 1. What you need

| Requirement | Details |
|---|---|
| OS | Windows 10 or 11 (64-bit) |
| GPU | **NVIDIA GPU strongly recommended** (GTX 1660 or better, RTX is best) |
| RAM | 16 GB recommended |
| Disk | About 8 GB free (Python packages and models) |
| Webcam | Any USB or built-in camera |

Without an NVIDIA GPU the app runs on CPU at only a few frames per second. That is too slow for a live call.

---

## 2. Install the prerequisites (one time)

### 2.1 Python 3.10
Download **Python 3.10.x** from https://www.python.org/downloads/windows/ and run the installer.

- Tick **"Add python.exe to PATH"**.
- Choose "Customize installation" and make sure **tcl/tk and IDLE** is ticked. The app's window needs it.
- It must be **3.10**. The bundled insightface package is built for 3.10 only.

Check it worked:
```
python --version
```
It should print `Python 3.10.x`.

### 2.2 ffmpeg
Download a Windows build from https://www.gyan.dev/ffmpeg/builds/ (the "release essentials" zip), unzip it, and add its `bin` folder to your PATH. Check:
```
ffmpeg -version
```

### 2.3 NVIDIA driver, CUDA and cuDNN (NVIDIA GPU users)
1. Update your NVIDIA driver from https://www.nvidia.com/drivers
2. Install **CUDA Toolkit 12.x** from https://developer.nvidia.com/cuda-downloads
3. Install **cuDNN 9 for CUDA 12** from https://developer.nvidia.com/cudnn

Check your GPU is visible:
```
nvidia-smi
```

### 2.4 OBS Studio (for video calls)
Install from https://obsproject.com

---

## 3. Install the app

1. Unzip the folder anywhere, for example `C:\DeepLiveCam`. Avoid very long paths and OneDrive folders.
2. Open the folder in File Explorer, type `powershell` in the address bar and press Enter.
3. Run the installer script:
```
.\setup.bat
```
This creates a virtual environment and installs every package. It takes 10-20 minutes.

### Doing it by hand instead
If you prefer to run the steps yourself, or `setup.bat` fails, run these one at a time:
```
python -m venv venv
```
```
venv\Scripts\activate
```
```
pip install insightface-0.7.3-cp310-cp310-win_amd64.whl
```
```
pip install -r requirements.txt
```
The insightface wheel must go **first**. Without it, pip tries to compile insightface and fails unless Visual Studio build tools are installed.

---

## 4. Get the models

Create a `models` folder in the project if it does not exist, then download:

**Swap model, fast version (recommended):**
```
curl -L -o models/inswapper_128_fp16.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
```

The slower full-precision model (`inswapper_128.onnx`, about 550 MB) downloads itself on first launch if the fast one is missing. Face detection models (`buffalo_l`) also download automatically on first launch.

**Optional face enhancer:** copy `gfpgan-1024.onnx` into `models` if your colleague sent it to you.

---

## 5. Run the app

```
.\run_app.bat
```
Or by hand:
```
venv\Scripts\activate
```
```
python run.py --execution-provider cuda
```

The first launch takes 1-2 minutes while models load and the GPU warms up. After that it starts quickly.

Without an NVIDIA GPU, run `python run.py` instead.

---

## 6. Use it in a video call

1. In the app, click **Select a face** and choose a clear, front-facing photo (a passport-style photo works well).
2. Click **Live**. A Preview window opens with your swapped face.
3. **Resize the Preview window to at least 1280x720** so OBS captures it sharp.
4. Open **OBS Studio**:
   - Under **Sources**, click **+** and choose **Window Capture**.
   - For Window, choose `[python.exe]: Preview`.
   - Right-click the source, choose **Transform**, then **Fit to Screen**.
   - In **Settings > Video**, set Base and Output resolution to **1280x720** and FPS to **30**.
5. Click **Start Virtual Camera** in OBS.
6. In Teams, Zoom or Meet, choose **OBS Virtual Camera** as your camera.

### Recommended settings in the app
| Setting | Value |
|---|---|
| Transparency | 100% (all the way right) |
| Sharpness | 40-60% |
| Mouth Mask | 30-40 (keeps your real lips moving naturally) |
| Eyes Mask | 0-25 (lower is faster; turn it off if the video lags) |
| Face Enhancer | None (sharper, but slows the video a lot) |

---

## 7. Troubleshooting

**The app is very slow or choppy**
Check that the GPU is actually in use:
```
venv\Scripts\python.exe -c "import onnxruntime as o; print(o.get_available_providers())"
```
The list should include `CUDAExecutionProvider`. If it does not, or you see cuDNN errors, the CUDA and cuDNN versions do not match the installed `onnxruntime-gpu`. Fix it one of two ways:
- Install CUDA 12 and cuDNN 9 (section 2.3), or
- Switch to the older runtime, which needs CUDA 11.8 and cuDNN 8:
```
pip install onnxruntime-gpu==1.17.0
```

**`No module named tkinter`**
Reinstall Python and tick the tcl/tk option (section 2.1).

**Camera not found or black picture**
Close any other app using the webcam (Teams, Zoom, OBS preview). Then pick the camera again in the app's **Select Camera** list.

**Lips flicker or the face jumps**
Lower the Eyes Mask slider to 0 to gain frame rate, and keep your face well lit and facing the camera.

**`pip install` fails on insightface**
You skipped the wheel step. Run `pip install insightface-0.7.3-cp310-cp310-win_amd64.whl` first, then `pip install -r requirements.txt` again.

**The first launch seems stuck**
It is downloading models and compiling GPU kernels. Wait up to 2 minutes.

---

## 8. Responsible use

Only use a face you have permission to use, and be open with the people on your call that you are using a face swap. Do not use it to impersonate or deceive anyone.
