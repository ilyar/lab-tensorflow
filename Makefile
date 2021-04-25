
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
export TF_CPP_MIN_LOG_LEVEL=3

venv:
	python3 -m venv venv
	. venv/bin/activate; if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

freeze: venv
	. venv/bin/activate; pip freeze > requirements.txt

test:\
src/weather\
src/mnist

clean:
	rm -rf venv
	rm -rf build

src/%: venv
	mkdir -p build
	. venv/bin/activate; python src/$*.py

mnist: src/mnist
weather: src/weather
