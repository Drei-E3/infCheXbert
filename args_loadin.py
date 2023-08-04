import pandas as pd

class Args_loadin():
    def __init__(self,args_dict):
        self.args = pd.Series(args_dict)
        self.dataset = self.args[['train_path',
                                  'dev_path',
                                  'test_path']]
        self.model_args = self.args[['pretrained_model_path',
                                     'output_model_path',
                                     'encoder',
                                     'feedforward_size',
                                     'bidirectional',
                                     'heads_num']]
        self.hyperpars = self.args[['seed',
                                    'epochs_num',
                                    'learning_rate',
                                    'hidden_size',
                                    'layers_num',
                                    'pooling',
                                    'dropout',
                                    'warmup',
                                    'batch_size']]
        self.tokenizer = self.args[['tokenizer',
                                    'vocab_path',
                                    'emb_size',
                                    'seq_length']]
        self.func = self.args[['config_path',
                               'mean_reciprocal_rank',
                               'workers_num',
                               'kernel_size',
                                'block_size',
                                'report_steps']]
        self.kg = self.args[['kg_name','no_vm']]
        self.subword= self.args[['subword_type','sub_vocab_path','subencoder']]

    def all_args(self):
        return self.args.keys()
    
    def type_of_args(self):
        return ['dataset','model_args','hyperpars','tokenizer','func','kg','subword']
    
    def pars_of(self,type_of_args):
        return self.type_of_args
    
    def parse(self):
        return self.args
    
    def parse_to_dict(self):
        return self.args.to_dict()

    def __str__(self) -> str:
        return 'use .args or .parse as arguments for the programm, their types are panadas.Serie(). use .info() to check all parameters'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def info():
        print('methodes:')
        print('all_args: return all names of arguments')
        print('type_of_args: all arguments are devided into different categories. use this method to check it')