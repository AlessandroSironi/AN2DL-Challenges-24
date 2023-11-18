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

| Model Name 		           | Accuracy | Precision | Recall | F1 Score |
| ---------------------------- | -------- | --------- | ------ | -------- |
| CNN_1      			       | 0.5800   | 0.4688    | 0.7895 | 0.5882   |
| CNN_2_Dropout      		   | 0.4300   | 0.3855    | 0.8421 | 0.5289   |
| CNN_2_Dropout_Sanitized      | 0.6200   | 0.5000    | 0.5526 | 0.5250   |
| CNN_3_Preprocessing          | 0.4200   | 0.3913    | 0.9473 | 0.5538   |
| CNN_3_Preprocessing_2        | 0.0000   | 0.0000    | 0.0000 | 0.0000   |
| CNN_4_Preprocessing_3        | 0.4100   | 0.3917    | 1.0000 | 0.5629   |
| CNN_4_Preprocessing_4        | 0.4200   | 0.3936    | 0.9736 | 0.5606   |
| CNN_6_efficientNet_2	       | 0.8300	  | 0.7692    | 0.7894 | 0.7792	  |

## 03/11/2023 - CNN_1
CNN_1 -> copy/paste from Labs to make it work.

## 05/11/2023 - CNN_2_Dropout
Added dropout layer, but made things worse. Sanitized version works best, but probably because of the sanitization steps only. 

## 06/11/2023 - CNN_3_Preprocessing
Aggressive preprocessing made things worse. 
Attempt _2 for a more balanced workflow. WAS NOT SUBMITTED

## 07/11/2023 - CNN_4_Preprocessing_2-4
Attempt at preprocessing gone wrong, AGAIN >:(
My life is sad AF regarding all aspects of my life. 

## 10/11/2023 - CNN_6_efficientNet_2
Good scores, implemented efficientNet M. Has augmentation. File is CNN_Isma_10nov.zip on OneDrive as the model became too large for GitHub.

