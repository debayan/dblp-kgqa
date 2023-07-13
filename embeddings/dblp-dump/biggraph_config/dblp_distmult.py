def get_torchbiggraph_config():

    config = dict(
        # I/O data
        entity_path='dblp_biggraph_data/',
        edge_paths=[],
        checkpoint_path='dblp_biggraph_models_diagonal/',

        # Graph structure
        entities={
            'all': {'num_partitions': 1},
        },
        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'diagonal',
        }],
        dynamic_relations=True,

        # Scoring model
        dimension=200,
        global_emb=False,
        comparator='dot',

        # Training
        num_epochs=10,
        num_edge_chunks=30,
        batch_size=40000,
        num_batch_negs=500,
        num_uniform_negs=500,
        loss_fn='softmax',
        lr=0.1,
        relation_lr=0.01,

        # Evaluation during training
        eval_fraction=0.001,
        eval_num_batch_negs=10000,
        eval_num_uniform_negs=0,

        # Misc
        verbose=1,
        num_gpus=2
    )

    return config
