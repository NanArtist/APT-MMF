import os, json, pickle, torch, joblib, random, datetime, argparse
import numpy as np
from sklearn.metrics import f1_score


###### ----- Path----- ######

analysis_file_data = os.path.join('data', 'embed')
ckpt_data = os.path.join('data', 'ckpt')
log_data = os.path.join('data', 'log')


###### ----- Schema & Label----- ######

NODE_TYPE = ['APT_Report', 'Registry', 'FilePath', 'FileName', 'Email', 'URL', 'Domain', 'IP', 'Tactic', 'Technique', 'Vulnerability', 'Malware']

NodeTypeAttr = {
    'Tactic':['ATT&CK_ID', 'name', 'description'],
    'Technique': ['ATT&CK_ID', 'name', 'description', 'associated_tactic'],
    'Vulnerability': ['CVE_ID', 'description'],
    'Malware':['hash', 'avclass_BEH', 'avclass_CLASS', 'avclass_FAM', 'avclass_FILE', 'imphash', 'pe_resource', 'pe_resource_lang', 'tags'],
    'Domain': ['domain', 'malicious_category'],
    'URL': ['url'],
    'IP': ['IP_address', 'geolocation'],
    'FilePath': ['filepath'],
    'Email': ['email'],
    'Registry': ['registry'],
    'FileName': ['filename'],
}

index_apt = {
    0: 'APT29', 
    1: 'APT32', 
    2: 'APT33', 
    3: 'APT34', 
    4: 'APT37', 
    5: 'BITTER', 
    6: 'Cobalt', 
    7: 'Confucius', 
    8: 'DarkHotel', 
    9: 'FIN6', 
    10: 'FIN7', 
    11: 'Kimsuky', 
    12: 'Lazarus', 
    13: 'MuddyWater', 
    14: 'ProjectSauron', 
    15: 'SideWinder', 
    16: 'Sofacy', 
    17: 'StrongPity', 
    18: 'TA505',
    19: 'TeamTNT', 
    20: 'Turla'
}


###### ----- Args ----- ######

default_configure = {
    'num_epochs': 500,       
    'patience': 30,          
    'cuda': True,       
    # Optimizer
    'lr': 0.005,
    'weight_decay': 0.001,
    # BERT finetune
    'ft_hidden_dim':256,
    'ft_out_dim':64,
    # Multilevel attention networks
    'emb_dim':256,                
    'hidden_dim':256,             
    'dropout_rate':0.80,          
    'num_heads': [8,[32]],        
    'hidden_units': 8,           
    'dropout': 0.40,             
}


def args_parse():
    parser = argparse.ArgumentParser('APT-MMF')
    parser.add_argument('-s', '--seed', type=int, default=72, help='Random seed')
    args = parser.parse_args().__dict__
    args = setup(args)
    return args


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['cuda'] = True if torch.cuda.is_available() else False
    return args


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


###### ----- Training ----- ######

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.time = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
        self.filename = os.path.join(ckpt_data, self.time+'_early_stop'+'.pth')
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))


###### ----- Evaluation ----- ######

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1


def evaluate(model, g, inputs, mask, labels=None, loss_func=None):
    model.eval()
    with torch.no_grad():
        logits = model(g, inputs)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    return loss, accuracy, micro_f1, macro_f1


def wrong_classification(labels, preds):
    row_indices = (labels!=preds).nonzero().flatten()
    report_id_label = np.genfromtxt(os.path.join(analysis_file_data, 'report_id_label.txt'), dtype=np.dtype(str), delimiter='\t', encoding='utf-8')
    index_apt = read_pickle(os.path.join(analysis_file_data, 'index_apt.pkl'))
    result = []
    for each in row_indices:
        result.append((int(report_id_label[int(each)][0]), index_apt[int(labels[each])], index_apt[int(preds[each])]))
    return result


###### ----- I/O Tools ----- ######

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    return json_dict


def write_json(result_dict, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)


def read_pickle(path, mode='pickle'):
    with open(path, 'rb') as f:
        if mode != 'joblib':
            result_dict = pickle.load(f)
        else:
            result_dict = joblib.load(f)
    return result_dict


def write_pickle(result_dict, path, mode='pickle'):
    with open(path, 'wb') as f:
        if mode != 'joblib':
            pickle.dump(result_dict, f)
        else:
            joblib.dump(result_dict, f, protocol = 4)


def write_log(logstr_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        for each in logstr_list:
            f.write(each+'\n')

