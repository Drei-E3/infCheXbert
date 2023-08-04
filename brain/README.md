
## information of kg.csv

***columns name:***

- 'relation', 'display_relation', 

- 'x_index', 'x_id', 'x_type', 'x_name', 'x_source',

- 'y_index', 'y_id', 'y_type', 'y_name', 'y_source'.

***types and numbers of knowledge (based on x_index):***
|  types             |   numbers |
|---                 |---        |
| drug               |   2805696 |
| gene/protein       |   2631229 |
| anatomy            |   1566154 |
| disease            |    341244 |
| effect/phenotype   |    257096 |
| biological_process |    252202 |
| molecular_function |     96723 |
| cellular_component |     93102 |
| pathway            |     47716 |
| exposure           |      9336 |

for more details, see in [kg_split.ipynb Script](kg_split.ipynb).


## SPO file and outputs of [kg_split.ipynb Script](kg_split.ipynb)

Spo stands for Subject Predicate Object triples, which should be separated by ```'\t'``` and without index. For convenience, it should be save in the folder ```'./kgs'```


**[kg_split.ipynb Script](kg_split.ipynb) offers 2 spo files outputs.**

- [CheXpert_KG.spo](kgs/CheXpert_KG.spo) include knowledge about the labels in MIMIC CXR and CheXpert dataset and anatomical knowledge along with medical effect/phenotype as well. there are 38251 entries in total

- [kg_anatomy3kAndAtelectasis.spo ](kgs/kg_anatomy3kAndAtelectasis.spo) is only used for debug and performs terriblly during the experiment. It includes knowledge about Atelectasis, and relation of drug_effect, of disease_phenotype_positive, and of phenotype_phenotype. there are 3136 entries in total.
