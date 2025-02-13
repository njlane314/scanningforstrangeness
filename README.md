# scanningforstrangeness

scp nlane@uboonegpvm05.fnal.gov:/exp/uboone/app/users/nlane/production/KaonShortProduction04/srcs/ubana/ubana/searchingforstrangeness/training_output.root .

/gluster/home/niclane/miniforge3/envs/pythondl/bin/python image.py

## Noether Setup Instructions

These instructions are to help setting up Miniconda and the correct Python environment for training Deep Learning CNN models on the Noether cluster.

### Prerequisites

#### Kerberos Setup

To transfer files between `uboonegpvm` and Noether, Kerberos is needed. Follow these steps:

1. Download the **krb5.conf** for SL7 from [here](https://authentication.fnal.gov/krb5conf/).
2. Copy the file into your home directory on Noether.
3. Once downloaded, get a Kerberos ticket on Noether by running the following command:

    ```bash
    kinit -fA <username>@FNAL.GOV
    ```

#### Requesting a GPU Session and Installing Miniforge

After logging into the Noether cluster, you will need to create a grid session with GPU access and install Miniforge:

1. Request a GPU session by running:

    ```bash
    qrsh request_gpus=1
    ```

2. Navigate to your home directory and install Miniforge:

    ```bash
    cd
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

3. Once Miniforge is installed, initialise conda:

    ```bash
    source ~/.bashrc
    conda init
    ```

### Python Environment Setup

Next, you will create the required Python environment and install the necessary libraries:

1. Create a new conda environment:

    ```bash
    conda create -n python3LEE python=3.8
    ```

2. Activate the environment:

    ```bash
    conda activate python3LEE
    ```

3. Install the required libraries:

    ```bash
    conda install scipy pandas==1.0.5 matplotlib pyyaml tqdm scikit-learn jupyter
    pip install torch opencv-python
    ```

4. Verify that PyTorch is detecting the GPU:

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

### Jupyter Notebook Setup

To run Jupyter notebooks on Noether with port forwarding, you will need to create three scripts.

1. **Jupyter Script**: Create the script `/gluster/home/<username>/scripts/bin/jupyter.sh` to launch Jupyter:

    ```bash
    #!/bin/bash
    # Script to run a Jupyter notebook on a specific port
    source /gluster/home/<username>/bin/conda_setup.sh python3LEE
    alias converttopy='jupyter nbconvert --to script'
    alias openjpnotebook='jupyter notebook --no-browser --port 1234'
    cd /gluster/home/<username>/DeepLearning/
    jupyter notebook --no-browser --port=1234
    ```

2. **Condor Submission File**: Create a Condor submission file to run the Jupyter notebook with resource allocation. Save this as `/gluster/home/<username>/etc/jupyter.sub`:

    ```bash
    executable      = bin/jupyter.sh
    request_memory  = 8G
    request_cpus    = 4
    request_gpus    = 1
    request_disk    = 5G
    initialdir      = $ENV(HOME)/scripts
    output          = out/jupyter/jupyter-$(Process).out
    error           = out/jupyter/jupyter-$(Process).err
    log             = out/jupyter/jupyter-$(Process).log
    arguments       = $(Process)
    should_transfer_files = yes
    when_to_transfer_output = ON_EXIT
    queue 1
    ```

3. **Conda Environment Setup Script**: Create a script to manage your Conda environments. Save this as `/gluster/home/<username>/bin/conda_setup.sh`:

    ```bash
    #!/bin/bash
    /gluster/home/<username>/miniforge3/etc/profile.d/conda.sh
    conda init
    source ~/.bashrc
    conda activate "$1"
    alias converttopy='jupyter nbconvert --to script'
    alias openjpnotebook='jupyter notebook --no-browser --port 1234'
    ```

### Port Forwarding and Connecting to Jupyter

To access Jupyter notebooks remotely, follow these steps:

1. Disconnect from Noether and reconnect with port forwarding:

    ```bash
    ssh -gL 1234:localhost:1234 <username>@noether.hep.manchester.ac.uk
    ```

2. Submit the Condor job to start the Jupyter notebook:

    ```bash
    cd scripts
    condor_submit etc/jupyter.sub
    ```

3. Connect to the running Condor job using:

    ```bash
    condor_ssh_to_job -ssh "ssh -gL 1234:localhost:1234" JOB_ID
    ```

4. Open the Jupyter notebook by visiting `http://localhost:1234` in your browser. Your working directory will be `/gluster/home/<username>/DeepLearning/`.
