### Attribution/emb
This folder is recommended for storing input data files of the programs. The *load_cti_kg()* function in *Main.py* can be used to load data from the files in this folder, and the implementation of *load_cti_kg()* needs to be defined by users according to their datasets. However, the output interfaces of *load_cti_kg()*, which can be seen as the inputs of our framework, are dataset-independent and well-designed as follows:

- labels, num_classes:

  *labels* is a PyTorch tensor with the shape (num_report) and each of its elements is the label number for the corresponding report. *num_classes* is the number of the label types(i.e. the number of APT groups).

- train_mask, val_mask, test_mask:

  The three PyTorch tensors have the same shape: (num_report). Each element of these tensors is either 1 or 0. A value of 1 indicates the corresponding report belongs to the train/validation/test set, while 0 indicates it does not.

- attribute_type_feat, nlt_feat, topo_relation_feat:

  The three PyTorch tensors represent the node features of three modalities. *attribute_type_feat* represents attribute type features with a shape of (num_node, 64), *nlt_feat* represents natural language text features with a shape of (num_node, bert_embed_dim), and *topo_relation_feat* represents topological relationship features with a shape of (num_node, 128). More details can be found in Section 3.3 of our paper.

- node_type_vec, heterG_adj, report_node:

  *node_type_vec* is a PyTorch tensor with a shape of (num_node, num_ioc_type) and each row of the tensor is a one-hot vector representing the ioc type of the corresponding node. *heterG_adj* and *report_node* are two PyTorch tensors with the same shape: (num_report, num_node). Each row of *heterG_adj* records the adjacency between the corresponding report node and its first-order and second-order IOC neighbors. Each row of *report_node* is a one-hot vector representing the corresponding report node within all nodes. The three tensors are used for the IOC type-level attention in our method.

- homoG_adj_MPs:

  *homoG_adj_MPs* is a list of DGL graphs and each of them records the adjacency of APT report nodes based on a specific metapath. These graphs can be seen as the homogeneous graphs generated from the metapaths. The length of the list is 20 (i.e. the number of the metapaths in Table 5 of our paper).
