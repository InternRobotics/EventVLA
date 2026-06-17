# X-VLA

## Environment Preparation

### 1️⃣ Installation

```bash
# Clone the repository
cd policy/X-VLA

# Create and activate Conda environment
conda create -n XVLA python=3.10 -y
conda activate XVLA

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Download X-VLA-Base-Model

```bash
huggingface-cli download --repo-type model 2toINF/X-VLA-Pt --local-dir ./checkpoints/X-VLA-Pt
```


## Training

### 1️⃣ Data Preparation

The RoboTwin-Mem dataset downloaded from Hugging Face needs to have a **language_instruction** key added to the HDF5 files in order to be compatible with X-VLA training. To do this, modify the **data_dir** and **instructions_dir** fields in hdf5_add_language_instruction.py, and then run:

```bash
python hdf5_add_language_instruction.py
```
This will add the required target key to the dataset.


### 2️⃣ Modify 'meta.json' File

Modify all file paths in the **datalist** to your target directory; no other changes are required.

### 3️⃣ Start Training

Modify the **models**, **train_metas_path**, and **output_dir** fields in **train.sh** so that they match your actual paths. 

Specifically，
- **models** refers to the path of the previously downloaded X-VLA-Pt
- **train_metas_path** refers to the path of meta.json
- **output_dir** refers to the directory where checkpoints will be saved.

Then, run

```bash
bash train.sh
```

## Evaluation

**Attention:** The saved checkpoint directory may only contain 'config.json', 'model.safetensors', 'state.json'. Please copy all the other files in 'X-VLA-Pt' to the target checkpoint directory (don't overwrite 'config.json', 'model.safetensors' and 'state.json' files).

### 1️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):

```bash
conda activate X-VLA
python -m deploy --port 4567 --model_path X-VLA-RoboTwin-Mem # change to your own model_path
```

### 2️⃣ Run the Client Evaluation

The evaluation client in `evaluation/RoboTwin-Mem/client.py` automatically locates the RoboTwin-Mem repository root and loads language instructions from `description/task_instruction`.

Modify the **task_name** in `evaluation/RoboTwin-Mem/eval_robotwin_mem.sh` to your target task name. The name should be one of the RoboTwin-Mem task slugs listed in `ALL_TASKS`.

Launch the RoboTwin-Mem evaluation client to connect to your X-VLA server:

```bash
# reopen a clean terminal

cd evaluation/RoboTwin-Mem

conda activate RoboTwin-Mem

bash eval_robotwin_mem.sh
```
