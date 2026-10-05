# USER MANUAL & OPERATOR GUIDE
## SIH26169 — AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals

**Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)**  
**Version:** 1.0 (Production Release)  
**System Platform:** Windows 10/11 (Python 3.10+ or Standalone Executable)

---

## 1. System Requirements & Installation

### 1.1 Prerequisites
- **Operating System:** Windows 10/11 (x64)
- **Processor:** Intel Core i5 / AMD Ryzen 5 or higher
- **RAM:** Minimum 4 GB (8 GB recommended)
- **Python Version:** Python 3.10 to 3.14 (if running from source)
- **Display:** 1080p ($1920 \times 1080$) recommended for full Mission Control UI layout

### 1.2 Running the Standalone Executable
If using the compiled standalone build:
1. Navigate to `C:\Users\Lenovo\Desktop\sih 2nd proj\`.
2. Double-click `SIH_FSOC_Tracker.exe` to launch the Desktop Mission Control GUI.
3. No Python installation or external dependencies are required.

### 1.3 Running from Source Code
1. Open PowerShell or Command Prompt in the project root directory:
   ```powershell
   cd "C:\Users\Lenovo\Desktop\sih 2nd proj"
   ```
2. Install required Python packages:
   ```powershell
   python -m pip install torch numpy opencv-python scipy matplotlib pillow pytest
   ```
3. Run the automated test suite to verify system integrity:
   ```powershell
   python -m pytest tests/ -v
   ```

---

## 2. Quick Start Guide & Execution Modes

The unified entry point `main.py` supports four distinct operational modes:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **Interactive GUI (Default)** | `python main.py` or `python main.py --gui` | Launches full Mission Control HUD and interactive control station. |
| **Automated Benchmark** | `python main.py --benchmark` | Runs full scientific benchmark sweep, generating plots and Markdown reports in `reports/`. |
| **Headless Simulation** | `python main.py --headless --duration 10` | Executes headless digital twin at maximum compute speed, logging metrics to terminal. |
| **Live Webcam Tracking** | `python main.py --webcam 0` | Ingests live optical feed from USB/webcam (device 0) and tracks physical beacons. |
| **Video File Tracking** | `python main.py --video path/to/video.mp4` | Ingests pre-recorded optical video and tracks beacons in real-time. |
| **Custom Configuration** | `python main.py --config config.json` | Loads custom physical, optical, or controller parameters from a JSON file. |

---

## 3. Mission Control Desktop GUI Tour

```
+--------------------------------------------------------------------------------------------------+
| SIH26169: AI-BASED VIRTUAL CAMERA TRACKING TESTBED (COARSE ALIGNMENT)   ISRO / SAC Telemetry     |
+--------------------------------------------------------------------+-----------------------------+
|                                                                    | [LIVE TELEMETRY SCORECARD]  |
|                                                                    | STATE: TRACK                |
|                    VIEWPORT CANVAS (1040x585)                      | Loop Rate: 41.2 FPS (24 ms) |
|                                                                    | Tracking Error: 13.5 px     |
|                       + Boresight Crosshair (Center)               | Lock Retention: 99.2%       |
|                                                                    | Gimbal Pose: Pan, Tilt      |
|             ( ) 2-sigma Kalman Ellipse (Cyan)                      +-----------------------------+
|              *  Ground Truth (Green Circle)                        | [OPTICAL FEED SOURCE]       |
|              #  Detected Centroid (Blue Diamond)                   | Stream: [Digital Twin   v]  |
|             --- Gating Boundary (Yellow Dashed)                    +-----------------------------+
|                                                                    | [DISTURBANCE CONTROLS]      |
|                                                                    | Turbulence (Cn2 proxy): --- |
|                                                                    | Micro-Vibration Amp:    --- |
|                                                                    | Sensor Noise Sigma:     --- |
|                                                                    | [TRAJECTORY & CONTROL]      |
|                                                                    | Motion: [figure8        v]  |
|                                                                    | Control:[pid_cov_aware  v]  |
|                                                                    | [ACTION BUTTONS]            |
|                                                                    | [⚡ Inject Occlusion (1.5s)] |
|                                                                    | [🔄 Reset Tracker & Gimbal]  |
|                                                                    | [📊 Run Benchmark Suite]    |
+--------------------------------------------------------------------+-----------------------------+
```

### 3.1 Live Sensor Viewport & HUD Visual Elements

The left viewport displays the sensor focal plane with overlaid real-time computer vision and tracking symbology:
1. **White Reticle (Crosshair & Circle):** The optical boresight center $(c_x, c_y) = (640, 360)$. The goal of coarse alignment is to center the beacon inside this reticle.
2. **Green Circle:** True ground truth beacon location (available in digital twin mode).
3. **Blue Diamond:** Detected optical centroid computed via intensity-weighted moments.
4. **Cyan Ellipse:** 2$\sigma$ ($95.4\%$) Kalman Filter position uncertainty ellipse. The ellipse shrinks as velocity converges and expands during occlusions.
5. **Yellow Dashed Boundary:** Mahalanobis gating limit ($\chi_2^2 = 9.21$). Any detection outside this boundary is rejected as celestial clutter or a decoy.
6. **Orange Spiral Path:** Active Archimedean search path displayed when the state machine is in `SEARCH` mode.

### 3.2 Live Telemetry Scorecard

- **STATE Badge:** Displays current state machine state:
  - `SEARCH` (Red): Sweeping uncertainty cone.
  - `CANDIDATE` (Yellow): Verifying persistent detection ($M$-of-$N$ check).
  - `TRACK` (Green): Nominal closed-loop locked tracking.
  - `COAST` (Orange): Target occluded; dead-reckoning trajectory extrapolation.
  - `REACQUIRE` (Red): Search corridor along projected velocity vector.
- **Loop Rate & Latency:** Real-time frame execution rate (FPS) and end-to-end processing latency (ms).
- **Tracking Error:** Instantaneous error in pixels and milli-radians ($\text{mrad}$), with cumulative Root Mean Square Error (RMSE).
- **Lock Retention Rate:** Percentage of total elapsed frames in `TRACK` state ($R_{\text{lock}} = N_{\text{track}} / N_{\text{total}}$).
- **Gimbal Pose:** Instantaneous Pan and Tilt orientation of the 2-axis gimbal mount in degrees.

---

## 4. Interactive Disturbance Sliders & Action Controls

Operators can dynamically stress-test the tracking system in real time using the right-hand control panel:

### 4.1 Disturbance Sliders
1. **Atmospheric Turbulence ($C_n^2$ proxy, Range: $0.0 \to 1.0$):**
   - Directly controls Angle-of-Arrival (AoA) stochastic wander, log-normal scintillation fading, and thermal beam PSF blurring.
   - *Test Procedure:* Slide to $0.8$–$1.0$ to observe severe beam distortion and verify that the Kalman filter maintains lock.
2. **Platform Micro-Vibration (Amplitude, Range: $0.0^\circ \to 0.25^\circ$):**
   - Injects high-frequency multi-harmonic structural vibrations ($12\text{ Hz}, 29\text{ Hz}, 47\text{ Hz}$) into the gimbal base.
3. **Sensor Gaussian Noise ($\sigma$, Range: $0.0 \to 15.0$):**
   - Simulates sensor thermal readout noise and pixel degradation.

### 4.2 Interactive Action Buttons
- **⚡ Inject Occlusion (1.5s):** Spawns an opaque cloud obscuring the optical beacon for $1.5\text{ seconds}$.
  - *Expected Behavior:* The system transitions immediately from `TRACK` to `COAST`. The cyan uncertainty ellipse expands as the filter propagates dead-reckoning velocity. Upon cloud clearance, the beacon is instantly re-locked into `TRACK` without dropping into `SEARCH`.
- **🔄 Reset Tracker & Gimbal:** Clears all filter covariances, resets the gimbal to $(0^\circ, 0^\circ)$, and initializes a fresh search cycle.
- **📊 Run Automated Benchmark Suite:** Executes the full reproducible benchmark suite in a background thread and opens the resulting evaluation summary upon completion.

---

## 5. Optical Feed Source Selection (Hardware-Ready Mode)

The system includes a dedicated **Optical Feed Source** selector:
1. **Simulation (Digital Twin):** Uses the internal physics and disturbance simulation engine.
2. **Live Webcam (Device 0):** Connects to the host USB/integrated camera, processes real optical video frames, and commands the virtual gimbal tracker in real time.
3. **Recorded Video File...:** Opens a file dialog allowing the operator to load any MP4, AVI, or MOV optical beacon video.

---

## 6. Telemetry Logging & Data Formats

When running headless simulations or benchmarks, telemetry is exported to the specified `--output-dir` (default: `reports/`):

### 6.1 CSV Telemetry Format (`telemetry.csv`)
Columns:
- `frame`: Frame index (integer)
- `sim_time_sec`: Elapsed simulation time in seconds
- `state`: Track state (`SEARCH`, `CANDIDATE`, `TRACK`, `COAST`, `REACQUIRE`)
- `fps`: Rolling loop rate in frames per second
- `latency_ms`: Total frame processing time in milliseconds
- `error_px`: Distance from optical boresight to target in pixels
- `error_mrad`: Distance from optical boresight to target in milliradians
- `cam_pan_deg`: Gimbal pan angle in degrees
- `cam_tilt_deg`: Gimbal tilt angle in degrees
- `kf_pan_deg`: Kalman estimated inertial pan angle
- `kf_tilt_deg`: Kalman estimated inertial tilt angle
- `kf_vel_pan_deg_s`: Kalman estimated pan velocity
- `kf_vel_tilt_deg_s`: Kalman estimated tilt velocity

### 6.2 JSON Summary Format (`summary.json`)
Contains key performance figures:
```json
{
  "fps": 41.2,
  "frame_latency_ms": 24.27,
  "acquisition_time_sec": 0.033,
  "lock_retention_rate_pct": 99.2,
  "rmse_error_px": 13.58,
  "rmse_error_mrad": 1.854,
  "reacquisition_time_sec": 0.0,
  "stage_latencies_ms": {
    "simulation": 18.2,
    "detection": 4.1,
    "estimation": 0.8,
    "control": 0.2
  }
}
```

---

## 7. Troubleshooting & FAQ

**Q1: The GUI window opens but the video canvas is black.**  
*Fix:* Verify that the Optical Feed Source is set to "Simulation (Digital Twin)". If in Webcam mode, verify that device 0 is connected and not in use by another application.

**Q2: PyTorch warning regarding `weights_only=False` appears in console.**  
*Status:* Expected behavior in PyTorch 2.6+ when loading neural network checkpoints containing custom NumPy normalization arrays. Handled cleanly in `perception/discriminator.py`.

**Q3: Can I run the benchmark in a purely non-graphical environment (e.g. CI/CD or Linux cluster)?**  
*Yes:* Run `python main.py --benchmark --output-dir reports/ci_run`. It requires no display server and outputs all plots and Markdown summaries automatically.

**Q4: How do I change the default camera resolution or FOV?**  
*Fix:* Create a custom JSON configuration file (e.g. `my_config.json`) modifying the `optical` dictionary:
```json
{
  "optical": {
    "image_width": 1920,
    "image_height": 1080,
    "fov_horizontal_deg": 15.0,
    "fov_vertical_deg": 8.44
  }
}
```
Run `python main.py --config my_config.json`. All geometry and projection matrices will scale automatically.

---
*End of User Manual.*
