# Layout Quality Quantifier (LQ2)

The repository provides supplementary materials for ACM-SIGCHI 2021 submission:  *Learning to Automate Chart Layout Configurations Using Crowdsourced Paired Comparison*.

## Getting Started

Our code is run in the `Python 3` environment. In particular, some rely on `Jupiter Notebook`. You might need to use `python3` or `pip3` depending on your configurations.

```shell
cd LQ2
pip3 install -r requirements.txt
```

## Overview

- src: source code for the ranking network and the baseline approaches
- dataset: the dataset used in MTurk studies
  - You may directly download the entire vega-lite json specifications, the corresponding images, and the graphical features from the baseline [[Google Drive](https://drive.google.com/drive/folders/1g3IlT0l_0r1oP7kcm_tXfhYniz8KC1gW?usp=sharing)].
- mturk: code for generating MTurk charts and analyzing results
- user-study: evaluate the application (compared with Human, Default, and Random)
