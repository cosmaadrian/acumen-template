# Acumen Deep Learning Research Project ✨ Template ✨

Before starting any deep-learning project, use this template to make your life as a researcher easier.

Keep your sanity! Work with `wandb` (included by default)!

### Open Source projects that used **✨ The Template ✨**

-  [It’s Just a Matter of Time: Detecting Depression with Time-Enriched Multimodal Transformers](https://github.com/cosmaadrian/time-enriched-multimodal-depression-detection)
-  [Learning Gait Representations with Noisy Multi-Task Learning](https://github.com/cosmaadrian/gaitformer)
-  [Exploring Self-Supervised Vision Transformers for Gait Recognition in the Wild](https://github.com/cosmaadrian/gait-vit)
-  [Reading Between the Frames: Multi-Modal Depression Detection in Videos from Non-Verbal Cues]()

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

### Adding classes for Datasets, Models, Trainers and Evaluators using `🔧Forge`

Datasets, Models and Trainers must be placed in the appropriate directory, added to their respective `__init__.py` file, and added to the `nomenclature.py` file.

### Structure for the configuration file.

You'll get the hang of it!

#### Extending configs.
Use `$extends$: <path>` in a config file if you want to extend it.

*configs/a.yaml*
```
foo: bar
```

*configs/b.yaml*
```
$extends$: configs/a.yaml
baz: zab
```
Running with this config file will result in ```args == {'foo': bar, 'baz': zab}```

#### Including other configs
Use ```$includes$: [<path1>, <path2>, ...]``` to include other configuration files directly.
*configs/a.yaml*
```
foo: bar
```

*configs/b.yaml*
```
$includes$:
	- configs/a.yaml
```
Running with this config file will result in ```args == {'foo': bar}```

#### Referencing other values
You can reference another value from the current configuration using the ```value = ${other_value}``` syntax.
```
a: b
c: ${a}
```
Running with this config file will result in ```args == {'a': b, 'c': b}```

### Update to the latest version

To keep your project updated with the latest `lib/` folder, just run the command:

```
	python lib/forge.py update
```

This command will override the lib folder with the latest changes. **WARNING. Any changes you made to `lib/` will be lost.**
