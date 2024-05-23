import os, time, torch
from model_utils import log_data, args_parse, EarlyStopping, score, evaluate, write_log

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def model_run(args):

    start = time.time()

    # load data
    homoG_adj_MPs,labels,num_classes,train_mask,val_mask,test_mask,heterG_adj,report_node,attribute_type_feat,nlt_feat,topo_relation_feat,node_type_vec = load_cti_kg()
    print('dataset loaded.')

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    heterG_adj = heterG_adj.to(args['device'])
    report_node = report_node.to(args['device'])
    attribute_type_feat = attribute_type_feat.to(args['device'])
    nlt_feat = nlt_feat.to(args['device'])
    topo_relation_feat = topo_relation_feat.to(args['device'])
    node_type_vec = node_type_vec.to(args['device'])

    homoG_adj_MPs = [graph.to(args['device']) for graph in homoG_adj_MPs]
    inputs = heterG_adj, report_node, attribute_type_feat, nlt_feat, topo_relation_feat, node_type_vec

    # load model
    model = Attribution(# Bert Finetune
                        nlt_in_size=nlt_feat.shape[1],
                        ft_out_dim=args['ft_out_dim'],
                        # Multilevel attention networks
                        emb_dim=args['emb_dim'],            # IOC type-level attention   
                        type_dim=node_type_vec.shape[1],    
                        dropout_ioc=args['dropout_ioc'],  
                        num_heads=args['num_heads'],        
                        num_meta_paths=len(homoG_adj_MPs),  # metapath-based neighbor node-level and metapath semantic-level attention
                        hidden_size=args['hidden_units'],   
                        out_size=num_classes,               
                        dropout_mpneigh=args['dropout_mpneigh'],            
                        # Others
                        cuda=args['cuda']).to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    log_strs = [str(args)]

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(homoG_adj_MPs, inputs)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, homoG_adj_MPs, inputs, val_mask, labels, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        log_str = 'Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | ' \
                'Val Loss {:.4f} | Val Acc {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}' \
                .format(epoch+1, loss.item(), train_acc, train_micro_f1, train_macro_f1, val_loss.item(), val_acc, val_micro_f1, val_macro_f1)
        log_strs.append(log_str)
        print(log_str)

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, homoG_adj_MPs, inputs, test_mask, labels, loss_fcn)
    log_str = 'Test loss {:.4f} | Test Acc {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_acc, test_micro_f1, test_macro_f1)
    log_strs.append(log_str)
    print(log_str)

    end = time.time()
    log_str = 'Total seconds for train and test: {}s'.format(int(end-start))
    log_strs.append(log_str)
    print(log_str)

    write_log(log_strs, os.path.join(log_data, stopper.time+'_log.txt'))
    return None


if __name__ == '__main__':

    args = args_parse()
    model_run(args)

