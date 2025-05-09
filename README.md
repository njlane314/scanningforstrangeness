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

## HTCondor Commands
The following HTCondor commands are used in this project:

- `condor_submit`: Submits jobs to the HTCondor queue. For example:
`condor_submit -i getenv=True` starts an interactive shell on a compute node, used for setting up or starting the VS Code tunnel.
- `condor_submit etc/deprecated.sub` submits a job using a deprecated submit file, likely for legacy or testing purposes (not recommended for current use unless specified).
- `condor_q`: Displays information about jobs in the HTCondor job queue. This command allows users to monitor the status of their submitted jobs, such as whether they are running, idle, or held.
- `condor_tail`: Views the last part of a file in the sandbox of a running job. For example, `condor_tail -f -maxbytes 100000 16033` follows the output of job 16033 in real-time, displaying up to 100,000 bytes of the output file. This is useful for monitoring job progress or debugging.
- `condor_rm`: Remove job, i.e. `condor_tail 16033`

## Requirements File

The following `requirements.txt` file lists the dependencies for this project:

```text
anyio @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_anyio_1742243108/work
argon2-cffi @ file:///home/conda/feedstock_root/build_artifacts/argon2-cffi_1733311059102/work
argon2-cffi-bindings @ file:///home/conda/feedstock_root/build_artifacts/argon2-cffi-bindings_1725356560642/work
arrow @ file:///home/conda/feedstock_root/build_artifacts/arrow_1733584251875/work
asttokens @ file:///home/conda/feedstock_root/build_artifacts/asttokens_1733250440834/work
async-lru @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_async-lru_1742153708/work
attrs @ file:///home/conda/feedstock_root/build_artifacts/attrs_1741918516150/work
awkward @ file:///home/conda/feedstock_root/build_artifacts/awkward_1742480894434/work
awkward_cpp @ file:///home/conda/feedstock_root/build_artifacts/awkward-cpp_1742473351834/work
babel @ file:///home/conda/feedstock_root/build_artifacts/babel_1738490167835/work
beautifulsoup4 @ file:///home/conda/feedstock_root/build_artifacts/beautifulsoup4_1738740337718/work
bleach @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_bleach_1737382993/work
Brotli @ file:///home/conda/feedstock_root/build_artifacts/brotli-split_1725267488082/work
cached-property @ file:///home/conda/feedstock_root/build_artifacts/cached_property_1615209429212/work
certifi @ file:///home/conda/feedstock_root/build_artifacts/certifi_1739515848642/work/certifi
cffi @ file:///home/conda/feedstock_root/build_artifacts/cffi_1725571112467/work
charset-normalizer @ file:///home/conda/feedstock_root/build_artifacts/charset-normalizer_1735929714516/work
colorama @ file:///home/conda/feedstock_root/build_artifacts/colorama_1733218098505/work
comm @ file:///home/conda/feedstock_root/build_artifacts/comm_1733502965406/work
cramjam @ file:///home/conda/feedstock_root/build_artifacts/cramjam_1726116418982/work
debugpy @ file:///home/conda/feedstock_root/build_artifacts/debugpy_1741148409996/work
decorator @ file:///home/conda/feedstock_root/build_artifacts/decorator_1740384970518/work
defusedxml @ file:///home/conda/feedstock_root/build_artifacts/defusedxml_1615232257335/work
Deprecated @ file:///opt/conda/conda-bld/deprecated_1659726428456/work
exceptiongroup @ file:///home/conda/feedstock_root/build_artifacts/exceptiongroup_1733208806608/work
executing @ file:///home/conda/feedstock_root/build_artifacts/executing_1733569351617/work
fastjsonschema @ file:///home/conda/feedstock_root/build_artifacts/python-fastjsonschema_1733235979760/work/dist
filelock @ file:///home/conda/feedstock_root/build_artifacts/filelock_1741969488311/work
fqdn @ file:///home/conda/feedstock_root/build_artifacts/fqdn_1733327382592/work/dist
fsspec @ file:///home/conda/feedstock_root/build_artifacts/fsspec_1741403990995/work
gmpy2 @ file:///home/conda/feedstock_root/build_artifacts/gmpy2_1733462549337/work
h11 @ file:///home/conda/feedstock_root/build_artifacts/h11_1733327467879/work
h2 @ file:///home/conda/feedstock_root/build_artifacts/h2_1738578511449/work
hpack @ file:///home/conda/feedstock_root/build_artifacts/hpack_1737618293087/work
httpcore @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_httpcore_1731707562/work
httpx @ file:///home/conda/feedstock_root/build_artifacts/httpx_1733663348460/work
hyperframe @ file:///home/conda/feedstock_root/build_artifacts/hyperframe_1737618333194/work
idna @ file:///home/conda/feedstock_root/build_artifacts/idna_1733211830134/work
importlib_metadata @ file:///home/conda/feedstock_root/build_artifacts/importlib-metadata_1737420181517/work
importlib_resources @ file:///home/conda/feedstock_root/build_artifacts/importlib_resources_1736252299705/work
ipykernel @ file:///home/conda/feedstock_root/build_artifacts/ipykernel_1719845459717/work
ipyparallel @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_ipyparallel_1741003660/work
ipython @ file:///home/conda/feedstock_root/build_artifacts/ipython_1701831663892/work
isoduration @ file:///home/conda/feedstock_root/build_artifacts/isoduration_1733493628631/work/dist
jedi @ file:///home/conda/feedstock_root/build_artifacts/jedi_1733300866624/work
Jinja2 @ file:///home/conda/feedstock_root/build_artifacts/jinja2_1741263328855/work
joblib @ file:///home/conda/feedstock_root/build_artifacts/joblib_1733736026804/work
json5 @ file:///home/conda/feedstock_root/build_artifacts/json5_1733272076743/work
jsonpointer @ file:///home/conda/feedstock_root/build_artifacts/jsonpointer_1725302957584/work
jsonschema @ file:///home/conda/feedstock_root/build_artifacts/jsonschema_1733472696581/work
jsonschema-specifications @ file:///tmp/tmpk0f344m9/src
jupyter-events @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_jupyter_events_1738765986/work
jupyter-lsp @ file:///home/conda/feedstock_root/build_artifacts/jupyter-lsp-meta_1733492907176/work/jupyter-lsp
jupyter_client @ file:///home/conda/feedstock_root/build_artifacts/jupyter_client_1733440914442/work
jupyter_core @ file:///home/conda/feedstock_root/build_artifacts/jupyter_core_1727163409502/work
jupyter_server @ file:///home/conda/feedstock_root/build_artifacts/jupyter_server_1734702637701/work
jupyter_server_terminals @ file:///home/conda/feedstock_root/build_artifacts/jupyter_server_terminals_1733427956852/work
jupyterlab @ file:///home/conda/feedstock_root/build_artifacts/jupyterlab_1741964057182/work
jupyterlab_pygments @ file:///home/conda/feedstock_root/build_artifacts/jupyterlab_pygments_1733328101776/work
jupyterlab_server @ file:///home/conda/feedstock_root/build_artifacts/jupyterlab_server_1733599573484/work
MarkupSafe @ file:///home/conda/feedstock_root/build_artifacts/markupsafe_1733219680183/work
matplotlib-inline @ file:///home/conda/feedstock_root/build_artifacts/matplotlib-inline_1733416936468/work
metakernel @ file:///home/conda/feedstock_root/build_artifacts/metakernel_1734293641883/work
mistune @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_mistune_1742402716/work
mpmath @ file:///home/conda/feedstock_root/build_artifacts/mpmath_1733302684489/work
nbclient @ file:///home/conda/feedstock_root/build_artifacts/nbclient_1734628800805/work
nbconvert @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_nbconvert-core_1738067871/work
nbformat @ file:///home/conda/feedstock_root/build_artifacts/nbformat_1733402752141/work
nest_asyncio @ file:///home/conda/feedstock_root/build_artifacts/nest-asyncio_1733325553580/work
networkx @ file:///home/conda/feedstock_root/build_artifacts/networkx_1698504735452/work
notebook @ file:///home/conda/feedstock_root/build_artifacts/notebook_1741968175534/work
notebook_shim @ file:///home/conda/feedstock_root/build_artifacts/notebook-shim_1733408315203/work
numpy @ file:///home/conda/feedstock_root/build_artifacts/numpy_1732314280888/work/dist/numpy-2.0.2-cp39-cp39-linux_x86_64.whl#sha256=62d98eb3da9f13e6b227c430d01026b7427f341b3fdcb838430f2a9e520417b1
opentelemetry-api @ file:///home/conda/feedstock_root/build_artifacts/opentelemetry-api_1742553828384/work
overrides @ file:///home/conda/feedstock_root/build_artifacts/overrides_1734587627321/work
packaging @ file:///home/conda/feedstock_root/build_artifacts/packaging_1733203243479/work
pandocfilters @ file:///home/conda/feedstock_root/build_artifacts/pandocfilters_1631603243851/work
parso @ file:///home/conda/feedstock_root/build_artifacts/parso_1733271261340/work
pexpect @ file:///home/conda/feedstock_root/build_artifacts/pexpect_1733301927746/work
pickleshare @ file:///home/conda/feedstock_root/build_artifacts/pickleshare_1733327343728/work
pillow @ file:///home/conda/feedstock_root/build_artifacts/pillow_1735929703139/work
pkgutil_resolve_name @ file:///home/conda/feedstock_root/build_artifacts/pkgutil-resolve-name_1733344503739/work
platformdirs @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_platformdirs_1742485085/work
portalocker @ file:///home/conda/feedstock_root/build_artifacts/portalocker_1731964072963/work
prometheus_client @ file:///home/conda/feedstock_root/build_artifacts/prometheus_client_1733327310477/work
prompt_toolkit @ file:///home/conda/feedstock_root/build_artifacts/prompt-toolkit_1737453357274/work
psutil @ file:///home/conda/feedstock_root/build_artifacts/psutil_1740663125313/work
ptyprocess @ file:///home/conda/feedstock_root/build_artifacts/ptyprocess_1733302279685/work/dist/ptyprocess-0.7.0-py2.py3-none-any.whl#sha256=92c32ff62b5fd8cf325bec5ab90d7be3d2a8ca8c8a3813ff487a8d2002630d1f
pure_eval @ file:///home/conda/feedstock_root/build_artifacts/pure_eval_1733569405015/work
pycparser @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_pycparser_1733195786/work
Pygments @ file:///home/conda/feedstock_root/build_artifacts/pygments_1736243443484/work
PySocks @ file:///home/conda/feedstock_root/build_artifacts/pysocks_1733217236728/work
python-dateutil @ file:///home/conda/feedstock_root/build_artifacts/python-dateutil_1733215673016/work
python-json-logger @ file:///home/conda/feedstock_root/build_artifacts/python-json-logger_1677079630776/work
pytz @ file:///home/conda/feedstock_root/build_artifacts/pytz_1738317518727/work
PyYAML @ file:///home/conda/feedstock_root/build_artifacts/pyyaml_1737454647378/work
pyzmq @ file:///home/conda/feedstock_root/build_artifacts/pyzmq_1741805177758/work
referencing @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_referencing_1737836872/work
requests @ file:///home/conda/feedstock_root/build_artifacts/requests_1733217035951/work
rfc3339_validator @ file:///home/conda/feedstock_root/build_artifacts/rfc3339-validator_1733599910982/work
rfc3986-validator @ file:///home/conda/feedstock_root/build_artifacts/rfc3986-validator_1598024191506/work
rpds-py @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_rpds-py_1740153283/work
scikit-learn @ file:///home/conda/feedstock_root/build_artifacts/scikit-learn_1736496755362/work/dist/scikit_learn-1.6.1-cp39-cp39-linux_x86_64.whl#sha256=e8f978e37bb47e04e1337a63f75697b723d6d25f58e477734555faed033884ba
scipy @ file:///home/conda/feedstock_root/build_artifacts/scipy-split_1716470218293/work/dist/scipy-1.13.1-cp39-cp39-linux_x86_64.whl#sha256=e6696cb8683d94467891b7648e068a3970f6bc0a1b3c1aa7f9bc89458eafd2f0
Send2Trash @ file:///home/conda/feedstock_root/build_artifacts/send2trash_1733322040660/work
six @ file:///home/conda/feedstock_root/build_artifacts/six_1733380938961/work
sniffio @ file:///home/conda/feedstock_root/build_artifacts/sniffio_1733244044561/work
soupsieve @ file:///home/conda/feedstock_root/build_artifacts/soupsieve_1693929250441/work
-e git+https://github.com/facebookresearch/SparseConvNet.git@cf251d058959a9dbaccb25fc919dc4f4548be232#egg=sparseconvnet
stack_data @ file:///home/conda/feedstock_root/build_artifacts/stack_data_1733569443808/work
sympy @ file:///home/conda/feedstock_root/build_artifacts/sympy_1736248176451/work
terminado @ file:///home/conda/feedstock_root/build_artifacts/terminado_1710262609923/work
threadpoolctl @ file:///home/conda/feedstock_root/build_artifacts/threadpoolctl_1741878222898/work
tinycss2 @ file:///home/conda/feedstock_root/build_artifacts/tinycss2_1729802851396/work
tomli @ file:///home/conda/feedstock_root/build_artifacts/tomli_1733256695513/work
torch @ file:///croot/libtorch_1738269269729/work
torchaudio==2.5.1
torchvision==0.20.1
tornado @ file:///home/conda/feedstock_root/build_artifacts/tornado_1732615921868/work
tqdm @ file:///home/conda/feedstock_root/build_artifacts/tqdm_1735661334605/work
traitlets @ file:///home/conda/feedstock_root/build_artifacts/traitlets_1733367359838/work
triton==3.1.0
types-python-dateutil @ file:///home/conda/feedstock_root/build_artifacts/types-python-dateutil_1733612335562/work
typing_extensions @ file:///home/conda/feedstock_root/build_artifacts/typing_extensions_1733188668063/work
typing_utils @ file:///home/conda/feedstock_root/build_artifacts/typing_utils_1733331286120/work
uproot @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_uproot_1741681500/work
uri-template @ file:///home/conda/feedstock_root/build_artifacts/uri-template_1733323593477/work/dist
urllib3 @ file:///home/conda/feedstock_root/build_artifacts/urllib3_1734859416348/work
wcwidth @ file:///home/conda/feedstock_root/build_artifacts/wcwidth_1733231326287/work
webcolors @ file:///home/conda/feedstock_root/build_artifacts/webcolors_1733359735138/work
webencodings @ file:///home/conda/feedstock_root/build_artifacts/webencodings_1733236011802/work
websocket-client @ file:///home/conda/feedstock_root/build_artifacts/websocket-client_1733157342724/work
wrapt @ file:///croot/wrapt_1736540904746/work
xrootd @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_xrootd_1742590268/work/build-dir/bindings/python
xxhash @ file:///home/conda/feedstock_root/build_artifacts/python-xxhash_1740594800419/work
zipp @ file:///home/conda/feedstock_root/build_artifacts/zipp_1732827521216/work
zstandard==0.23.0
```
There are some requirements missing here, update locally firt

To update `requirements.txt` with additional packages, install them in the environment and run:
```bash
pip freeze > requirements.txt
```
