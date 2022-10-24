import inspect
import sys
import os
import requests
import random

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

"""
	python lib/forge.py list
"""

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

logo = "Forge🏗️"

class Command():
	def run(self, args = None):
		pass

class HelpCommand(Command):
	name = "help"
	description = f"Show {logo} help."

	def run(self, args = None):
		print(f"""
{logo} is a command line tool to help you to more easily manage your research project.

Get started with:
	{OKCYAN}python lib/forge.py list{ENDC}
""")

class ListCommand(Command):
	name = 'list'
	description = f"List all available commands for {logo}"

	def run(self, args = None):
		print(f"\n\n{logo} commands. Let's bulid that research paper!\n\n")

		for name, klass in sorted(commands.items(), key = lambda x: x[0]):
			print(f"{BOLD}{OKCYAN}{name} {ENDC}- {klass.description}")

		print("\n")

class ShowCommand(Command):
	name = "show"
	description = "Show the current available resources in this project."

	def run(self, args = None):
		if len(args):
			kinds = [a.upper() for a in args]
		else:
			kinds = ['MODELS', 'DATASETS', 'EVALUATORS', 'LOSSES', 'HEADS', 'TRAINERS']

		from lib import nomenclature

		for k in kinds:
			if k not in nomenclature.__dict__ or not len(nomenclature.__dict__[k]): continue

			print(f"{OKGREEN}::: {k}{ENDC}")
			for i, (name, resource) in enumerate(nomenclature.__dict__[k].items()):
				print(f"{OKCYAN}{i + 1}. {name} - {resource.__name__}{ENDC}")
			print("\n")

class EncourageCommand(Command):
	name = "encourage"
	description = "Stuck on a problem? Have some free motivation!"

	def run(self, args):
		quotes = requests.get('https://gist.githubusercontent.com/robatron/a66acc0eed3835119817/raw/0e216f8b6036b82de5fdd93526e1d496d8e1b412/quotes.txt')
		print('\n')
		print(f"{BOLD}{WARNING}" + random.choice(quotes.text.split('\n')) + ENDC)
		print('\n')

class CreateCommand(Command):
	name = "create"
	description = "Create a resource for this project. Creates a class with boilerplate, adds it to __init__.py and nomenclature.py"

	def run(self, args):
		print("Not implemented yet.")

commands = {obj.name: obj for name, obj in inspect.getmembers(sys.modules[__name__]) if name.endswith('Command') and name != 'Command'}

if __name__ == '__main__':
	if len(sys.argv) < 2:
		HelpCommand().run()
		exit(0)

	command_name = sys.argv[1]
	commands[command_name]().run(sys.argv[2:])
