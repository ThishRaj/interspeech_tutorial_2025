# Interspeech 2025
## Confidence Estimation for Trustworthy and Efficient Speech Systems (Slot 1)

Presenters - 
> Dr. Vipul Arora - Professor, Dept. of Electrical Engineering, P K Kelkar fellow, IIT Kanpur, India

> Dr. Nagarathna Ravi - Senior Scientist, CSIR - 4PI, Bangalore, India

> Mr. Thishyan Raj T - mvaak AI, SIIC - IIT Kanpur, India

**Abstract** - Estimating uncertainty in outputs can enhance the trustworthiness of AI systems. A calibrated classification model, apart from giving an output, gives a confidence value in its output that approximates the expected accuracy of the model for that output. A calibrated regression model, on the other hand, gives precise confidence intervals of the output. State-of-the-art (SOTA) approaches focus on calibrating model outputs or developing auxiliary models to estimate the confidence in model predictions. Two main types of uncertainty are aleatoric, i.e., arising from the stochastic mapping between the input and the output, and epistemic, stemming from the limitations in the model’s knowledge. Estimating them separately opens up further opportunities. Trustworthy speech systems based on confidence estimation can enhance various tasks, such as automatic speech recognition (ASR), speech enhancement and speaker diarization. The confidence estimates can help in decision making, enhancing performance and active learning, thereby making learning efficient in low-resource and atypical settings. This tutorial will cover the theoretical foundations and SOTA methods for uncertainty estimation and confidence calibration. This will include a deeper dive into confidence calibration for end-to-end ASR. Finally, we will present some applications of confidence calibration for low-resource ASR and music analysis.

### Pre-requisites for the Tutorial

To run locally, access to a computer system with an Nvidia GPU with atleast 4GB of GPU memory is necessary to get started (running on CPU is possible but slow).
Alternatively the files can be uploaded to a Google colab notebook for the tutorial.

#### Steps to setup NeMo (ignore if running on google colab)

Please download and install the Nvidia NeMo toolkit. The tutorial will primarly be conducted using this toolkit. Pre-requisite softwares/packages that need to be installed are - `miniconda`, `sox`, `libsndfile1` and `ffmpeg`. Follow this installation [link](https://www.anaconda.com/docs/getting-started/miniconda/install) to set up `miniconda`. The other libraries can be installed using -

```bash
sudo apt update
sudo apt install sox libsndfile1 ffmpeg

mkdir interspeech_2025_tutorial # Or any directory name to store all the data, codes and models that will be used in this tutorial.
conda create --name interspeech-demo python==3.10.12
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
conda activate interspeech-demo
pip install -e ./NeMo[all]
pip install jupyter torchaudio
cd ..
```

Next download and prepare the required dataset using the commands below from the project directory.

```bash
mkdir -p WORK_DIR/DATA

wget -O ./WORK_DIR/DATA/train_clean_100.tar.gz https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz # Optional step to train confidence models on Librispeech data
wget -O ./WORK_DIR/DATA/dev_clean.tar.gz https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz
wget -O ./WORK_DIR/DATA/dev_other.tar.gz https://openslr.magicdatatech.com/resources/12/dev-other.tar.gz
wget -O ./WORK_DIR/DATA/test_clean.tar.gz https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz
wget -O ./WORK_DIR/DATA/test_other.tar.gz https://openslr.magicdatatech.com/resources/12/test-other.tar.gz

python ./NeMo/scripts/dataset_processing/get_librispeech_data.py --data_root=./WORK_DIR/DATA --data_set="train_clean_100" # Optional step to train confidence models on Librispeech data
python ./NeMo/scripts/dataset_processing/get_librispeech_data.py --data_root=./WORK_DIR/DATA --data_set="dev_clean"
python ./NeMo/scripts/dataset_processing/get_librispeech_data.py --data_root=./WORK_DIR/DATA --data_set="dev_other"
python ./NeMo/scripts/dataset_processing/get_librispeech_data.py --data_root=./WORK_DIR/DATA --data_set="test_clean"
python ./NeMo/scripts/dataset_processing/get_librispeech_data.py --data_root=./WORK_DIR/DATA --data_set="test_other"
```

Finally to download the ASR model -

```bash
git clone <link to repo>

```

Training Codes and google colab notebook will be updated soon!