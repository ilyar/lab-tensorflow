
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
LOG_LEVEL=3

venv:
	pip install virtualenv
	virtualenv venv

install: requirements.installed
requirements.installed: venv requirements.txt
	. venv/bin/activate; pip install -Ur requirements.txt
	touch requirements.installed

freeze: venv
	. venv/bin/activate; pip freeze
	. venv/bin/activate; pip freeze > requirements.txt

test:\
quickstart

clean:
	rm -rf venv
	rm -rf requirements.installed

quickstart: install
	. venv/bin/activate; TF_CPP_MIN_LOG_LEVEL=${LOG_LEVEL} python src/quickstart.py
