# e-ViL

This repository contains the e-SNLI-VE dataset, the HTML files for the e-ViL human evaluation framework, and e-UG model of our ICCV 2021 paper:

*e-ViL: A Dataset and Benchmark for Natural Language Explanations in Vision-Language Tasks* ([ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Kayser_E-ViL_A_Dataset_and_Benchmark_for_Natural_Language_Explanations_in_ICCV_2021_paper.html)).

## e-SNLI-VE

The train, dev, and test splits are in the `data` folder. The `.csv` files contain Flickr30k Image ID's. Flickr30k can be downloaded [here](https://www.kaggle.com/hsankesara/flickr-image-dataset).

## e-ViL MTurk Questionnaires

The `e-ViL_MTurk` folder contains the MTurk questionnaires for e-SNLI-VE, VQA-X, and VCR. These `HTML` files can be uploaded to the Amazon Mechanical Turk platform for crowd-sourced, human evaluation.

## e-UG

e-UG uses UNITER as vision-language model and GPT-2 to generate explanations. The UNITER implementation is based on the code of the [Transformers-VQA](https://github.com/YIKUAN8/Transformers-VQA) repo and the GPT-2 implementation is based on [Marasovic et al. 2020](https://github.com/allenai/visual-reasoning-rationalization).

The entry point for training and testing the models is in `eUG.py`.

### Environment

The environment file is in `eUG.yml`.

Create the environment by running `conda env create -f eUG.yml`.

### COCOcaption package for automatic NLG metrics

In order to run NLG evaluation in this code you need to download the package from this [Google Drive link](https://drive.google.com/file/d/1nLlrtQlsP5kSeB9L0PTle4PeGL9CJg29/view?usp=sharing). It needs to be placed in the root directory of this project.

### Downloading the data

#### e-SNLI-VE

1. Run this [script](https://github.com/ChenRocks/UNITER/blob/master/scripts/download_ve.sh) to download the Faster-RCNN features for Flickr30k and store them in `data/esnlive/img_db/flickr30k/`.
3. Download the `.json` files, ready to be used with e-UG, from this [Google Drive link](https://drive.google.com/drive/folders/1ajL93SLltaKiBk2PgvaxCLAJSoXKAsZz?usp=sharing) and store them in `data/esnlive/`.

#### VQA-X

1. Download the Faster-RCNN features for MS COCO train2014 (17 GB) and val2014 (8 GB) images:
   
    ```
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/fasterRCNN_features
    unzip data/img/train2014_obj36.zip -d data/fasterRCNN_features && rm data/fasterRCNN_features/train2014_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/fasterRCNN_features
    unzip data/fasterRCNN_features/val2014_obj36.zip -d data && rm data/fasterRCNN_features/val2014_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data/fasterRCNN_features
    unzip data/fasterRCNN_features/test2015_obj36.zip -d data && rm data/fasterRCNN_features/test2015_obj36.zip
    ```

2. Download the VQA-X dataset from this [Google Drive link](https://drive.google.com/drive/folders/1zPexyNo_W8L-FYq6iPcERQ5cJUUJzYhl?usp=sharing) and store the splits in `data/vqax/`.   
#### VCR

1. Download the Faster R-CNN feature using this [script](https://github.com/ChenRocks/UNITER/blob/master/scripts/download_vcr.sh). 
2. Download the VCR `.json` files from this [Google Drive link](https://drive.google.com/drive/folders/1REopdRzF1tgik22LHf2i85MMLXjconQK?usp=sharing).
   
#### Pre-trained weights

Download the general pre-trained UNITER-base using this [link](https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-base.pt). The pre-trained UNITER-base for VCR is available from this [link](https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-base-vcr_2nd_stage.pt). We use the general pre-trained model for VQA-X and e-SNLI-VE, and the VCR pre-trained one for VCR.


### Training

Check the command line arguments in `param.py`.

Here is an example to train the model on e-SNLI-VE:

```
python eUG.py --task esnlive --train data/esnlive/esnlive_train.json --val data/esnlive/esnlive_dev.json --save_steps 5000 --output experiments/esnlive_run1/train
```

The model weights, Tensorboard logs, and a text log will be saved in the given output directory.

### Testing

Check the command line arguments in `param.py`.

Here is an example to test a trained model on the e-SNLI-VE test set:

```
python eUG.py --task esnlive --test data/esnlive/esnlive_test.json --load_trained experiments/esnlive_run1/train/best_global.pth --output experiments/esnlive_run1/eval 
```

All generated explanations, automatic NLG scores, and a text log will be saved in the given output directory.

## Citation

If you use this dataset or the e-ViL benchmark in your work, please cite our paper:

```
@InProceedings{Kayser_2021_ICCV,
    author    = {Kayser, Maxime and Camburu, Oana-Maria and Salewski, Leonard and Emde, Cornelius and Do, Virginie and Akata, Zeynep and Lukasiewicz, Thomas},
    title     = {E-ViL: A Dataset and Benchmark for Natural Language Explanations in Vision-Language Tasks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {1244-1254}
}
```