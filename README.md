# scanningforstrangeness

## Setup Instructions

### First Time Setup
1. **Install the Microsoft `Remote - Tunnels` extension** (VS Code Desktop only):
   - Open the `Extensions` sidebar (Ctrl+Shift-X), search for `ms-vscode.remote-server`, and install it.
2. **Connect to Noether**:
   - Replace `username` with your own username:
     ```shell
     ssh username@noether.hep.manchester.ac.uk
     ```
3. **Get the VS Code Server CLI**:
   ```shell
   mkdir vscode && cd vscode
   curl -L 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
   tar -xf vscode_cli.tar.gz
   ```
4. **Start an interactive shell**:
   ```shell
   condor_submit -i getenv=True
   cd
   ```
5. **Set up the VS Code tunnel**:
   ```shell
   ./code tunnel
   ```
   - Follow the prompts: log in with your Microsoft account, enter the provided code on the website, sign in with University credentials, name the tunnel (e.g., `noether`), and ignore the browser URL prompt if using the desktop app.

### Starting an Already Configured Tunnel
1. **Connect to Noether**:
   ```shell
   ssh username@noether.hep.manchester.ac.uk
   ```
2. **Start an interactive shell**:
   ```shell
   condor_submit -i getenv=True
   cd
   ```
3. **Start the VS Code tunnel**:
   ```shell
   ./code tunnel
   ```
   - Ignore the browser URL prompt if using the desktop app.

### Copying Data Files
- To copy the necessary data files, use the following command (replace `nlane` with your username if needed):
  ```shell
  scp nlane@uboonegpvm05.fnal.gov:/exp/uboone/app/users/nlane/production/KaonShortProduction04/srcs/ubana/ubana/searchingforstrangeness/training_output.root .
  ```
- The dot (`.`) copies the file to your current directory. Ensure you are in the desired directory on Gluster, such as `/gluster/data/your_area/scanningforstrangeness/data/`, before running the command. This is where the data files will be stored in the Gluster file system.
- **Note:** Replace `'your_area'` with your actual area or username in the destination path.

### Installing Miniforge on Gluster
1. **Navigate to the installation directory**:
   ```shell
   cd /gluster/data/your_area
   ```
   - **Note:** Replace `'your_area'` with your actual area or username.
2. **Download the Miniforge installer**:
   ```shell
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   ```
3. **Run the installer**:
   ```shell
   bash Miniforge3-Linux-x86_64.sh
   ```
4. **Follow the prompts** to complete the installation.
5. **Update your shell configuration**:
   - Add the following line to your `~/.bashrc` or `~/.bash_profile`:
     ```shell
     source /gluster/data/your_area/miniforge3/etc/profile.d/conda.sh
     ```

### Setting Up the Conda Environment
1. **Create a Conda environment**:
   - Create an environment named `pythondl` with Python 3.8:
     ```bash
     conda create -n pythondl python=3.8
     ```
2. **Activate the environment**:
   ```bash
   conda activate pythondl
   ```
3. **Install dependencies**:
   - Use the provided `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

## Example Usage

### `example.sh`
This Bash script runs the `segmentation.py` script for plane 0:
- Changes to the project directory.
- Sources Conda setup.
- Activates the `pythondl` environment.
- Runs the Python script with the `--plane 0` argument.

```bash
#!/bin/bash
cd /gluster/home/niclane/scanningforstrangeness
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh 
conda activate pythondl
/gluster/home/niclane/miniforge3/envs/pythondl/bin/python -u segmentation.py --plane 0
```

### `example.sub`
This HTCondor submission file configures a job to run `segmentation.sh`:
- Specifies a vanilla universe.
- Requests 16GB memory, 4 CPUs, 1 GPU, and 20GB disk.
- Sets output, error, and log file paths.
- Transfers the `segmentation.py` file and enables IO proxy.

```text
universe        = vanilla
executable      = bin/segmentation.sh
request_memory  = 16G
request_cpus    = 4
request_gpus    = 1
request_disk    = 20G
initialdir      = /gluster/home/niclane/scripts
output          = out/segmentation/segmentation-$(Cluster).$(Process).out
error           = out/segmentation/segmentation-$(Cluster).$(Process).err
log             = out/segmentation/segmentation-$(Cluster).$(Process).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = /gluster/home/niclane/scanningforstrangeness/segmentation.py
+WantIOProxy    = True
queue 1
```

Modify these accordingly to your task and save in your `/scripts/` area.

### `segmentation.py`
This Python script implements a U-Net model for image segmentation:
- Uses `argparse` for command-line arguments (e.g., `--plane`, `--num-epochs`).
- Loads data from a ROOT file using `uproot`.
- Defines a `UNet` class with convolutional and transpose convolutional blocks.
- Trains the model with a custom dataset (`ImageDataset`) and computes metrics (precision, recall).
- Saves model checkpoints and training metrics.

Key dependencies: `torch`, `uproot`, `numpy`.

## Requirements File

The following `requirements.txt` file lists the dependencies for this project:

```text
torch==1.10.0
uproot==4.0.0
numpy==1.21.0
```

To update `requirements.txt` with additional packages, install them in the environment and run:
```bash
pip freeze > requirements.txt
```
