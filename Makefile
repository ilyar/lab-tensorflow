
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
export TF_CPP_MIN_LOG_LEVEL=3

venv-activate: venv
	. venv/bin/activate

venv:
	pip install virtualenv
	virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt

freeze: venv-activate
	pip freeze > requirements.txt

test:\
mnist

clean:
	rm -rf venv

src/%: venv-activate
	python src/$*.py

mnist: src/mnist
