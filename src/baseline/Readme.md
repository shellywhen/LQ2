# Readme

This folder contains code to calculate man-crafted features for scoring as the baseline for our approach. 

`parse_feature_from_vega.py`: generate graphical features for later scoring.

`generate_empirical_score.py`: analyze graphical features and compute empirical scores for each image.

`Empircial-Measure.ipynb`: evaluate the empirical scores on the MTurk dataset.

## Output Results
The results generated from the python script can be found on the [[Google Drive](https://drive.google.com/drive/folders/1g3IlT0l_0r1oP7kcm_tXfhYniz8KC1gW?usp=sharing)].

- Graphical features
  - exp1: `~/exp1/graphical_features.pkl`
  - exp2: `~/exp2/graphical_features.pkl`
- Scores: a table of image scores under each criteria.
  - exp1: `~/exp1/metrics.csv`
  - exp2: `~/exp2/metrics.csv`

## Features

The definition of each terms and model parameters take reference from [1].

Generally, we consider four aspects to evaluate charts in our context:

- White Space: white space ratio, spread, distance
- Scale: graphic size, graphic size variance, graphic size constraint
- Unity: group size variance, mean group distance
- Balance: x - (a)symmetry, y - (a)symmetry

|Spec|Default $\alpha$|#Equation|#Parameter|Weight|
|---|---|---|---|---|
|White Space Area|2|9|10|0.2|
|Spread|600|10|60|50|
|Distance|50|12|49|50|
|Graphic Size|2|15|5|50|
|Graphic Size Variance|200|7|16|50|
|Min Graphic Size|$\tau$=0.04|17|9|125|
|Group Size Var|1|29|37|2.5|
|Group Distance Min|200|7|16|250|
|X Symmetry|1|6|42, 46|200|
|X Asymmetry|1|6|43, 47|0.2|
|Y Symmetry|1|7|44, 48|0.2|
|Y Asymmetry|1|7|45, 49|0.2|


## Reference

[1]. O’Donovan, P., Agarwala, A., & Hertzmann, A. (2014). Learning layouts for single-pagegraphic designs. *IEEE transactions on visualization and computer graphics*, *20*(8), 1200-1213. [[DOI]](https://doi.org/10.1109/TVCG.2014.48 ) [[Project Page]](http://www.dgp.toronto.edu/~donovan/layout/)

