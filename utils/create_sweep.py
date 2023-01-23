import wandb


def create_sweep(**kwargs):
    sweep_config = {"method": "random"}
    parameters_dict = {
        "optimizer": {"values": ["adam", "nadam", "adamax"]},
        "activation": {"values": ["sigmoid", "relu", "tanh", "selu", "gelu"]},
        "kernel_size": {"values": [3,5,7,9,11]},
        'beta_1': {"values":[0.45  , 0.5625, 0.675 , 0.7875, 0.9   , 1.0125, 1.125 , 1.2375,
       1.35  ]},
        'beta_2':{"values":[0.984 , 0.9855, 0.987 , 0.9885, 0.99  , 0.9915, 0.993 , 0.9945,
       0.996 ]}
    }
    parameters_dict.update(
        {
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                "distribution": "q_uniform",
                "min": 0.0001,
                "max": 0.001,
                "q": 0.0001,
            },
            "batch_size": {
                # integers between 32 and 256
                # with evenly-distributed logarithms
                "distribution": "q_log_uniform_values",
                "q": 8,
                "min": 32,
                "max": 128,
            },
            "dropout": {
                "distribution": "q_log_uniform_values",
                "q": 0.01,
                "min": 0.01,
                "max": 0.8,
            },
             "epsilon": {
                "distribution": "q_log_uniform_values",
                "q": 1.25e-08,
                "min": 5.000e-08,
                "max": 1.500e-07,
            }
        }
    )
    parameters_dict.update({"epochs": {"value": 100}})
    sweep_config["parameters"] = parameters_dict
    return wandb.sweep(
        sweep=sweep_config, project=kwargs["project"], entity=kwargs["entity"]
    )
