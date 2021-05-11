# e-ViL

This repository contains the e-SNLI-VE dataset and the HTML files for the human evaluation framework of our paper: 

*e-ViL: A Dataset and Benchmark for Natural Language Explanations in Vision-Language Tasks* (https://arxiv.org/abs/2105.03761)

## e-SNLI-VE

The train, dev, and test splits are in the `data` folder. The `.csv` files contain Flickr30k Image ID's. Flickr30k can be downloaded [here](https://www.kaggle.com/hsankesara/flickr-image-dataset).

## e-ViL MTurk Questionnaires

The `e-ViL_MTurk` folder contains the MTurk questionnaires for e-SNLI-VE, VQA-X, and VCR. These `HTML` files can be uploaded to the Amazon Mechanical Turk platform for crowd-sourced, human evaluation.

## Citation

If you use this dataset or the e-ViL benchmark in your work, please cite our paper:

```
@misc{kayser2021evil,
      title={e-ViL: A Dataset and Benchmark for Natural Language Explanations in Vision-Language Tasks}, 
      author={Maxime Kayser and Oana-Maria Camburu and Leonard Salewski and Cornelius Emde and Virginie Do and Zeynep Akata and Thomas Lukasiewicz},
      year={2021},
      eprint={2105.03761},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```




