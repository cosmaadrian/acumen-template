# Acumen Deep Learning Research Template

Before starting any deep-learning project, use this template to make your life as a researcher easier.

Keep your sanity! Work with `wandb` (included by default)!

### Usage:

Install `cookiecutter` with:
```
conda install cookiecutter
```

Create directory for project structure with:

```
cookiecutter gh:cosmaadrian/acumen-template
```

### Main Philosophy

All deep learning projects have three main components: a dataset, a model, a training procedure and an evaluation procedure.

This project's goal is to provide a ready-made deep learning research project structure. This template / framework is *opinionated*, assumes the usage of `pytorch` and `wandb`.

It is heavily based on `.yaml` configuration files and command line arguments to run experiments in a declarative way. The only things that the researcher needs to focus their time and energy is "bussiness logic", which means the data processing / model training / model evaluation. The main training loop and experiment tracking is automatically done by the framework. The usual callbacks for model checkpointing, early stopping and logging are provided by default.


### Command line management: 🔧Forge

You can use the `lib/forge.py` command line tool to better manage your project. Get started:

```
	python lib/forge.py help
```

### Generating classes for Datasets, Models, Trainers and Evaluators using `🔧Forge`

Datasets, Models and Trainers must be placed in the appropriate directory, added to their respective `__init__.py` file, and added to the `nomenclature.py` file. This process can be time consuming and / or confusing at first.

To automatically generate such a boilerplate class use the `forge.py` script:

```
	python lib/forge.py create dataset:MyDatasetClassName:my-dataset-class_name
	python lib/forge.py create model:MyModelClassName:my-model-class_name
	python lib/forge.py create trainer:MyTrainerClassName:my-trainer-class_name
```

This will add boilerplate code in the respective directories, add class names to the `__init__.py` file and add snake-case names to the `nomenclature.py` file.


### Structure for the configuration file.

You'll get the hang of it!

Don't forget to use `$extends$: ...` in a config file if you want to extend it.

### Update to the latest version

To keep your project updated with the latest `lib/` folder, just run the command:

```
	python lib/forge.py update
```

This command will override the lib folder with the latest changes. **WARNING. Any changes you made to `lib/` will be lost.**
