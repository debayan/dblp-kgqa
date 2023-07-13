TransE
------

CUDA_VISIBLE_DEVICES=1,3 torchbiggraph_train biggraph_config/dblp_transe.py -p edge_paths=dblp_biggraph_data/train_partitioned/
CUDA_VISIBLE_DEVICES=1 torchbiggraph_export_to_tsv biggraph_config/dblp_transe.py --entities-output dblp_biggraph_data_transe/entity_embeddings.tsv --relation-types-output dblp_biggraph_data_transe/relation_types_parameters.tsv
