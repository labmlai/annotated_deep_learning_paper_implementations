clean: ## Clean
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

build: clean ## Build PIPy Package
	python setup.py sdist bdist_wheel

check-content: build  ## List contents of PIPy Package
	tar -tvf dist/*.tar.gz

check: build  ## Check PIPy Package
	twine check dist/*

upload: build  ## Upload PIPy Package
	twine upload dist/*

install:  ## Install from repo
	pip install -e .

uninstall: ## Uninstall
	pip uninstall labml_nn

docs: ## Render annotated HTML
	find ./docs/ -name "*.html" -type f -delete
	find ./docs/ -name "*.svg" -type f -delete
	python utils/sitemap.py
	python utils/diagrams.py
	cd labml_nn; pylit --remove_empty_sections --title_md -t ../../../pylit/templates/nn -d ../docs -w *

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: clean build check upload help docs
.DEFAULT_GOAL := help
