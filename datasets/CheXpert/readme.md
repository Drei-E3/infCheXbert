# dataset

## extract free text reports into csv file

1. download [mimic-cxr-2.0.0 dataset](https://physionet.org/content/mimic-cxr/2.0.0/) and [mimic-cxr-2.0.0-chexpert.csv](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) on https://physionet.org/
2. run [reports_extraction.ipynb](reports_extraction.ipynb) file to extract free text reports into a csv file ```report_without_labels.csv``` extrahie. notice there is no labels in this csv files yet.
3. run [reports_with_label.ipynb](reports_with_label.ipynb) script to add expert annatation labels for mimic-cxr and obtain another csv file: ```report_with_labels.csv```

## data split

In a free text report there are several parts of information. the most important are **impression** and **finding** section. 

run [data.ipynb](data.ipynb) to split reports into finding dataset and impression dataset and the split the data into train, test, validation parts as well. By reproduction just locate the paths in 'set Auguments' section in infCheXbert ipynb  script.
