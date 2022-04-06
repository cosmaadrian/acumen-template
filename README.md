# {{cookiecutter.project_name}}

{{cookiecutter.project_description}}

Directory Structure:

.
├── {{cookiecutter.project_slug}}
│   ├── callbacks
│   ├── configs
│   │   ├── base_config.yaml
│   │   └── env_config.yaml
│   ├── evaluate.py
│   ├── evaluators
│   │   ├── __init__.py
│   │   ├── base_evaluator.py
│   │   └── utils.py
│   ├── experiments
│   │   └── README.md
│   ├── loggers
│   │   ├── __init__.py
│   │   └── wandb_logger.py
│   ├── models
│   │   └── __init__.py
│   ├── notebooks
│   │   └── README.md
│   ├── particular_model_trainers
│   │   ├── __init__.py
│   │   ├── acumen_trainer.py
│   │   └── losses
│   ├── requirements.txt
│   ├── schedulers
│   │   ├── __init__.py
│   │   ├── lr_finder.py
│   │   ├── onecycle.py
│   │   └── __pycache__
│   ├── scripts
│   │   └── README.md
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── main.py
│   ├── nomenclature.py
│   └── utils.py
└── README.md
