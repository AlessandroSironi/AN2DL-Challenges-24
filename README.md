# AN2DL-Challenges-24
Repo of the Challenges for the course "Advanced Neural Networks and Deep Learning" @ PoliMi

# Test Set
NB: Creare la cartella con questo esatto path: 

	AN2DL-Challenges-24/Challenge\ 1/data/phase_1 

e mettere public_data.zip (limiti per push di dimensioni > 100MB)

# .py to .ipybn
```bash
pip3 install jupytext
python3 -m jupytext --to notebook model.py
```

# Changelog and history for the models

## Scores

| Model Name 		 | Accuracy | Precision | Recall | F1 Score |
| ------------------ | -------- | --------- | ------ | -------- |
| CNN_1      		 | 0.5800   | 0.4688    | 0.7895 | 0.5882   |
| CNN_2_Dropout      | 0.4300   | 0.3855    | 0.8421 | 0.5289   |

## 03/10/2023 - CNN_1
CNN_1 -> copy/paste from Labs to make it work.

## 05/10/2023 - CNN_2_Dropout
Added dropout layer, but made things worse