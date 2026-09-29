# SnapLab-AI --- Beginner's User Guide

A practical guide to installing, running, and using SnapLab-AI on
Windows or through Modal.

## 1. What is SnapLab-AI?

SnapLab-AI is an engineering-vision assistant for inspecting electronic
components in circuit and hardware images. Its project includes a
YOLO-based detector, component refinement/classification,
circuit-relationship and connection reasoning, and optional AI-assisted
engineering explanations.

**Important:** Results are predictions, not guaranteed measurements or a
substitute for checking a real circuit. Image quality, occlusion, glare,
crowded layouts, and crossing wires can affect results. Verify important
findings against the physical circuit, schematic, and component
datasheets.

## 2. Choose how to run it

  -----------------------------------------------------------------------
  Option                              When to use it
  ----------------------------------- -----------------------------------
  **Local Windows app**               Run the interactive Gradio
                                      application on your PC.

  **Modal-hosted app**                Open the deployed application
                                      through its web endpoint.

  **Snapdragon-specific inference**   Use the Qualcomm GenieX/QAIRT path
                                      on a compatible Snapdragon
                                      environment.
  -----------------------------------------------------------------------

The static project website, if deployed separately, is not the same as
the interactive Python application.

## 3. Important project files

  --------------------------------------------------------------------------------
  Path                                         Purpose
  -------------------------------------------- -----------------------------------
  `app.py`                                     Local application entry point

  `modal_app.py`                               Modal deployment entry point

  `requirements.txt`                           Python dependencies

  `vision/component_inspector_v2.py`           Main inspection interface and logic

  `vision/component_fusion_engine.py`          Detection fusion/refinement

  `vision/component_refinement_engine.py`      Component classifier/refinement

  `vision/connection_engine.py`                Connection inference

  `vision/connection_graph.py`                 Connection graph representation

  `vision/wire_association.py`                 Wire/component association

  `vision/vlm/`                                VLM prompt, runtime, and deployment
                                               configuration

  `models/best.pt`                             Main YOLO model

  `models/component_classifier/best.pt`        Component classifier weights

  `models/component_classifier/classes.json`   Class labels

  `assets/`                                    Images and UI assets

  `engineering_state.py`                       Engineering-state logic
  --------------------------------------------------------------------------------

Do not delete or rename these files unless you also update the paths and
imports in the code.

## 4. Run locally on Windows

### Step 1 --- Open the project directory

Open PowerShell and run:

``` powershell
cd C:\Users\purus\SnapLab-AI
```

Use your actual project path if it is different.

### Step 2 --- Activate the existing environment

If your project already has a working `.venv`:

``` powershell
.\.venv\Scripts\Activate.ps1
```

If you need to create it, the project has previously been used with
Python 3.10.11:

``` powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it for this terminal only:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 3 --- Install dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installation may take a while, especially for PyTorch and
computer-vision packages.

### Step 4 --- Verify model files

``` powershell
Test-Path .\models\best.pt
Test-Path .\models\component_classifier\best.pt
Test-Path .\models\component_classifier\classes.json
```

Each command should return `True`.

### Step 5 --- Start the app

``` powershell
python app.py
```

Open the local URL printed in the terminal. The app has commonly used:

``` text
http://127.0.0.1:7860
```

If the terminal prints another port, use that URL.

### Step 6 --- Stop the app

In the PowerShell window running the app, press `Ctrl + C`.

## 5. Use the inspection interface

The interface can change between versions, so follow the controls shown
in your running Gradio page.

1.  **Prepare an image.** Use a sharp, well-lit image with components
    and wires visible. Avoid glare, blur, and strong shadows.
2.  **Upload the image.** Select an image using the interface's image
    input.
3.  **Run inspection.** Click the inspection/detection control provided
    by the interface. The first run may take longer while models
    initialize.
4.  **Review the output.** Depending on enabled features, results may
    include annotated detections, component names and confidence values,
    refined labels, relationship/connection information, or an
    AI-generated explanation.
5.  **Check uncertain results.** Manually verify low-confidence labels,
    overlapping parts, wire crossings, and inferred connections. If
    needed, take a closer image and run the inspection again.

Start with a simple circuit image before trying crowded or complex
layouts.

## 6. Configure AI reasoning securely

The engineering-reasoning module can use a Groq-compatible or OpenAI API
key, depending on the configuration in `vision/openai_engine.py`.

For local use, configure the environment variable expected by your code.
Examples:

``` text
GROQ_API_KEY=your_key_here
```

or:

``` text
OPENAI_API_KEY=your_key_here
```

Check `vision/openai_engine.py` for the exact variable names and any
provider/model settings supported by your version.

**Keep credentials safe:** - Never commit `.env` or API keys to Git. -
Do not include keys in screenshots or bug reports. - If a key is
exposed, revoke it with the provider and replace it. - Check your
provider's usage limits and billing.

Without a configured API key, API-backed explanations may be
unavailable; other application features may still work, depending on the
code's fallback behavior.

## 7. Use the Modal-hosted app

The project uses `modal_app.py` as a separate deployment entry point.

### Step 1 --- Open the project

``` powershell
cd C:\Users\purus\SnapLab-AI
```

### Step 2 --- Activate the environment containing Modal

If you created `venv-modal`:

``` powershell
.\venv-modal\Scripts\Activate.ps1
```

If you use the existing `modal-env`, activate that instead. You do not
need two environments to deploy the same app.

### Step 3 --- Check Modal

``` powershell
modal --version
modal token status
```

Authenticate if Modal requests it.

### Step 4 --- Check the secret

The deployment configuration references a Modal Secret named
`snaplab-secrets`:

``` powershell
modal secret list
```

The secret should contain the provider variable your app needs, such as
`GROQ_API_KEY` or `OPENAI_API_KEY`. Never print or share secret values.

### Step 5 --- Deploy

``` powershell
modal deploy .\modal_app.py
```

Wait for deployment to finish, then open the web endpoint shown by
Modal.

### Step 6 --- Redeploy after changes

``` powershell
modal deploy .\modal_app.py
```

### Step 7 --- Monitor cloud usage

Check the Modal dashboard for logs, function activity, credits, and
usage. A deployed app can have zero live containers while idle; that
alone does not mean deployment failed.

**Cost note:** Cloud usage can consume credits or incur charges. Pricing
and balances can change. Avoid repeated test runs, and stop or remove
the app when you no longer need it.

## 8. Qualcomm / Snapdragon notes

The Qualcomm GenieX/QAIRT VLM path is configured for a compatible
Snapdragon X Elite environment and requires the appropriate device,
operating system, runtime, and model bundle.

A standard Modal Linux container does not provide the Snapdragon X Elite
NPU or its Windows-specific GenieX/QAIRT runtime. Therefore, do not
expect that Qualcomm-specific inference path to run on Modal. The hosted
app may still support its other features. Treat cloud CPU inference and
Snapdragon NPU inference as separate execution targets.

## 9. Troubleshooting

### `python` or `py` is not recognized

Install Python, reopen PowerShell, and check:

``` powershell
py --version
python --version
```

### The environment will not activate

Check that the activation script exists:

``` powershell
Test-Path .\.venv\Scripts\Activate.ps1
```

Run activation from the project directory.

### A model file is missing

``` powershell
Test-Path .\models\best.pt
Test-Path .\models\component_classifier\best.pt
Test-Path .\models\component_classifier\classes.json
```

Restore missing files or update the model paths in the code.

### The Gradio page will not open

-   Check that `python app.py` is still running.
-   Open the exact URL printed in the terminal.
-   Check whether the port is already in use.
-   Read the full terminal traceback.

### A Python module is missing

Activate the intended environment and run:

``` powershell
python -m pip install -r requirements.txt
```

If the problem persists, note the missing module and full traceback.

### AI reasoning is unavailable

-   Confirm the correct provider key is configured.
-   Check that the variable name matches the code.
-   Check network access and provider limits.
-   Never share the API key while troubleshooting.

### Modal deployment fails

-   Read the complete deployment error.
-   Confirm `modal_app.py`, `requirements.txt`, `assets/`, the model
    files, and `engineering_state.py` exist.
-   Confirm `snaplab-secrets` exists.
-   Check Modal function logs for runtime errors.
-   Fix the specific dependency, memory, or runtime issue rather than
    repeatedly redeploying.

### The Modal app deploys but fails on first visit

Deployment success does not guarantee successful application startup.
Open the endpoint, then inspect the function logs in the Modal
dashboard. Share relevant error text without keys or secrets.

## 10. Good practices

-   Keep backups of model weights and important project files.
-   Do not rename model files without updating code paths.
-   Keep API keys out of source control.
-   Use clear, focused circuit images.
-   Verify predictions before making hardware changes.
-   Debug local and Modal runs separately.
-   Check cloud usage before extended testing.
-   Record the model version and environment when comparing results.

## 11. Quick command reference

### Local run

``` powershell
cd C:\Users\purus\SnapLab-AI
.\.venv\Scripts\Activate.ps1
python app.py
```

### Modal deploy

``` powershell
cd C:\Users\purus\SnapLab-AI
.\venv-modal\Scripts\Activate.ps1
modal token status
modal secret list
modal deploy .\modal_app.py
```

### Check model files

``` powershell
Test-Path .\models\best.pt
Test-Path .\models\component_classifier\best.pt
Test-Path .\models\component_classifier\classes.json
```

## 12. Final checklist

-   [ ] Correct Python environment is active.
-   [ ] Dependencies are installed.
-   [ ] Required model files are present.
-   [ ] Local app or Modal endpoint opens.
-   [ ] Test image is clear and in focus.
-   [ ] Detections and inferred connections have been reviewed.
-   [ ] API credentials are stored securely.
-   [ ] Cloud usage is understood before extended testing.

**You are ready to begin.** Start with a clear image of a simple
circuit, review the detections, and then move on to more complex
layouts.
