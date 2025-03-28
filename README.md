[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14852686.svg)](https://doi.org/10.5281/zenodo.14852686)

# Towards a Holistic Framework for Multimodal LLM in 3D Brain CT Radiology Report Generation
<p align="center" width="100%">
<img src="overview.png"  width="100%" height="100%">
</p>
[Nature Communications Paper Link]((https://www.nature.com/articles/s41467-025-57426-0))


## Code
> this repository is modified from https://github.com/Luodian/Otter

```
git clone https://github.com/charlierabea/FORTE.git

cd FORTE
```

## Set-up Environment
```
# GPU with RAM > 36GB is recommended since our checkpoint is 32.54 GB
conda env create -f environment.yml
# Otter: Install the pytorch that matches your cuda version. (e.g. cuda 11.7 torch 2.0.0). We have successfully run this code on cuda 11.1 torch 1.10.1 and cuda 11.7 torch 2.0.0. You can refer to PyTorch's documentation, Latest or Previous.
conda activate forte
pip install torch
pip install -r requirements.txt
# You may encounter error in installing Hovorod if your local "cmake" version is too old. You can run the inference code without Hovorod installed.
```

## Generate reports
### Download Files
Place both the base model [luodian/OTTER-MPT7B-Init](https://huggingface.co/luodian/OTTER-MPT7B-Init/tree/main) and our instruction-tuned model hf folder [https://huggingface.co/Charliebear/BrainGPT] at **./checkpoint/** folder

Place the CQ500 external validation dataset (image file [https://drive.google.com/file/d/1iDLx7NqvTg8sBTVViQu5wq8OhPSovAo4/view?usp=drive_link] at the **./data/** folder
```
bash ./eval.sh
```
The output excel file will appear at **./Evaluation/pipeline/train/output**

## Evaluation
### Download Files
BrainGPT is fine-tuned based on Otter ([luodian/OTTER-MPT7B-Init](https://huggingface.co/luodian/OTTER-MPT7B-Init))

Please place the FORTE keyword file [[https://docs.google.com/spreadsheets/d/1NtlDOHDoVNa_xrypH5J79_5ZxL-5mPzM/edit?usp=sharing&ouid=104290583109385210784&rtpof=true&sd=true](https://drive.google.com/file/d/1cSa9KYhfXShe7hveNmNXKif9K6SArOE0/view?usp=drive_link)] at **./data/** folder

### 1. Automatic Evaluation
```
python3 Automatic_evaluation.py
```
***Please remove the Spice scorer at /pycocoevalcap/eval.py work before running the code. The Spice scorer is not included in the automatic evaluation code. This may avoid potential error and reduce run time.***

```
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]
```

### 2. Sentence pairing
```
python3 Sentence_pairing.py
```

### 3. FORTE Evaluation (Keyword lists for Brain CT/ Chest CT/ Abdomen CT/ Chest X-ray are provided)
```
python3 FORTE.py
```

### 4. Negation removal
```
python3 Negation_removal.py
```

## Reference

```
@article{Li2025,
  title={Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation},
  author={Li, Cheng-Yi and Chang, Kao-Jung and Yang, Cheng-Fu and Wu, Hsin-Yu and Chen, Wenting and Bansal, Hritik and Chen, Ling and Yang, Yi-Ping and Chen, Yu-Chun and Chen, Shih-Pin and Chen, Shih-Jen and Lirng, Jiing-Feng and Chang, Kai-Wei and Chiou, Shih-Hwa},
  journal={Nature Communications},
  volume={16},
  pages={2258},
  year={2025},
  doi={10.1038/s41467-025-57426-0}
}
```
Li, CY., Chang, KJ., Yang, CF. et al. Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation. Nat Commun 16, 2258 (2025). https://doi.org/10.1038/s41467-025-57426-0
