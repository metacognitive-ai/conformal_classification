# Makefile

.PHONY: build_venv
build_venv:
	test -d .venv || python3 -m venv .venv --prompt conformal_classification
	.venv/bin/pip install -U pip setuptools wheel
	.venv/bin/pip install -Ur requirements.txt

.PHONY: clean
clean:
	rm -rf .venv

# .PHONY: run
# run:
#	./run.sh

