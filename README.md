## Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This model uses [K-BERT](https://github.com/autoliuweijie/K-BERT) architecture and [UER](https://github.com/dbiir/UER-py) framework. In addition you can use it als multiclassification with multiple labels now. You can also use Pretrained models with transformer achitecture from huggingface just runing [convert_bert_from_huggingface_to_uer.py](convert_bert_from_huggingface_to_uer.py) 

```sh
python3 run convert_bert_from_huggingface_to_uer.py \
    # path of model from huggingface
    --input_model_path {./path_of_model_from_huggingface } \
    # path of model you want to put, which later would be use in infCheXbert_model.ipynb
    # better to save in ./models folder
    --output_model_path ./models/models_name.bin \
    # you would better to check layers by code model.state_dict() from torch. 
    # the standard layers would be 12 
    --layer_num 12
```

and after training translate the model in huggingface model by running [convert_bert_from_uer_to_huggingface.py](convert_bert_from_uer_to_huggingface.py) with

```sh
python3 run convert_bert_from_huggingface_to_uer.py \
    # path of model you have just trained, normally in outputs folders
    --input_model_path {./path_of_model_trained} \
    # any place you want to save
    --output_model_path {any place you want put} \
    # should consist with layers_num in arguments of infCheXbert_model.ipynb. default 12
    --layer_num 12 
```

## Brain(knowledge graphs):

the medical knowledge graphs used in this thesis consist of relation of labels () and anatomy medicine knowledge. they are formatted into a spo file which should be put into the folder ```./brain/kgs```. The file uses a medicine database created by [Precision Medicine Knowledge Graph (PrimeKG)](https://zitniklab.hms.harvard.edu/projects/PrimeKG/). The relative [article](https://www.biorxiv.org/content/10.1101/2022.05.01.489928v2) written by Payal Chandak*, Kexin Huang*, and Marinka Zitnik was public on Scientific Data 2023. 

The dataset is hosted on Harvard Dataverse, you can download it with this [link](https://dataverse.harvard.edu/api/access/datafile/6180620) and then run [kg_split.ipynb](brain/kg_split.ipynb) to manufacture the spo file. For more detail,see [readme](brain/README.md) file in the folder ```./brain```. and the original project in github [PrimeKG](https://github.com/mims-harvard/PrimeKG).

## 
