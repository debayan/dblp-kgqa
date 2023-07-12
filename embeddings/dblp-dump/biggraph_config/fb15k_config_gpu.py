#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="dblp_biggraph_data/",
        edge_paths=[
            "dblp_biggraph_data/train_partitioned",
            "dblp_biggraph_data/valid_partitioned",
            "dblp_biggraph_data/test_partitioned",
        ],
        checkpoint_path="model/dblp",
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=200,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=50,
        batch_size=5000,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        regularization_coef=1e-3,
        # Evaluation during training
        eval_fraction=0,
        # GPU
        num_gpus=1,
    )

    return config
