# FPL Optimization Tools

> **Based on** [open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver) by [fploptimised.com](https://fploptimized.com),
> licensed under the [Apache License 2.0](LICENSE).
> This is a personal fork with modifications by **Skye Fung** (2025).
> The original project remains the work of its authors; all credit for the core solver belongs to them.

---

This repository provides a set of tools for solving deterministic **Fantasy Premier League (FPL)** optimization problems.
The Python code uses **`pandas`** for data management, **`sasoptpy`** for building the optimization model, and **HiGHS** via **`highspy`** to solve the model.

It allows users to:

- Automatically select the best FPL squad based on the given projection data and solver settings.
- Customize squad constraints, formation rules, transfer strategies, and more.
- Modify data sources and parameters to suit personal models or preferences.

## 🔧 Installation

### 1. Install `uv`

`uv` handles **both Python installation and dependency management**, so you **do not need to install Python separately**.

**Windows (PowerShell)**

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installation, then verify:

```bash
uv --version
```

---

### 2. Install Git

**Windows**

Download from [git-scm.com](https://git-scm.com/download/win) and accept all default installation options.

**macOS**

Git is usually pre-installed. If not, run:

```bash
brew install git
```

### 3. Clone the Repository

Open a terminal (search for *Command Prompt* in Windows) and run:

```bash
cd Documents
git clone https://github.com/solioanalytics/open-fpl-solver.git
cd open-fpl-solver
```

### 4. Install Dependencies (and Python)
```bash
uv sync
```

## 🚀 Running the Optimizer

### 1. Add Projection Data

Place your projections file (e.g., `solio.csv`) in the `data/` folder.

### 2. Configure Data Source

If you are not using the default data source, update the `datasource` field in `data/user_settings.json` to match your CSV file name.

Example: if you are using a file named `projections.csv`, the settings file should contain:

```json
"datasource": "projections"
```

### 3. Edit Settings

Edit any desired settings in `comprehensive_settings.json` or `user_settings.json`.

- The majority of useful settings for most people will be in `user_settings.json`.
- `comprehensive_settings.json` provides a wider range of options that will be used as defaults unless altered in `user_settings.json`.

Details of what each setting does can be found in the `.md` file in the `/data/` folder.

### 4. Run the Solver

```bash
cd run
uv run python solve.py
```

## 🎥 Videos

There is a YouTube playlist [here](https://www.youtube.com/playlist?list=PLrIyJJU8_viOags1yudB_wyafRuTNs1Ed) by Sertalp, showing the early stages of this tool, explaining how it was built, and discussing ideas around optimization with a focus on FPL.

## 🌍 Browser-based optimization

There is also a browser-based version of the optimizer that doesn't require the download or installation of anything to your device, and works on mobile. It is hosted in a google colab notebook that can be found [here](https://colab.research.google.com/drive/1fwYcG28zpIOJf7R8yx31bDL_kJG1JRLu). Simply follow the instructions on that page to run the optimizer.

## 🛠️ Issues

If you have issues, feel free to open an issue on GitHub and I will get back to you as soon as possible.
Alternatively, you can email me at **chris.musson@hotmail.com**.
