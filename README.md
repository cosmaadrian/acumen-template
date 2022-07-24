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

All deep learning projects have three main components: a dataset, a model, a training procedure and an evaluation procedure. I have seen many students struggle with keeping their project code clean and organized and a have difficulty tracking experiments and organizing them.

This project's goal is to provide a ready-made deep learning research project structure. This template / framework is *opinionated*, assumes the usage of `pytorch` and `wandb`.

It is heavily based on `.yaml` configuration files and command line arguments to run experiments in a declarative way. The only things that the researcher needs to focus their time and energy is data cleaning / model training and evaluation. The main training loop and experiment tracking is automatically done by the framework.

### Generating classes for Datasets, Models, Trainers and Evaluators

Datasets, Models and Trainers must be placed in the appropriate directory, added to their respective `__init__.py` file, and added to the `nomenclature.py` file. This process can be time consuming and / or confusing at first. I have created a helper script to keep the hassle at a minimum.

To generate such a boilerplate class use the `forge.py` script:

```
	python forge.py --create dataset:MyDatasetClassName:my-dataset-class_name
	python forge.py --create model:MyModelClassName:my-model-class_name
	python forge.py --create trainer:MyTrainerClassName:my-trainer-class_name
```

This will add boilerplate code in the respective directories, add class names to the `__init__.py` file and add snake-case names to the `nomenclature.py` file.
