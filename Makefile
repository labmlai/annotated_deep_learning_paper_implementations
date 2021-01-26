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
	cd labml_nn; pylit --remove_empty_sections --title_md -t ../../../pylit/templates/nn -d ../docs -w *

pages-old: ## Copy to lab-ml site
	cd labml_nn; pylit --remove_empty_sections --title_md -t ../../../pylit/templates/nn_old -d ../html/labml_nn *
	@cd ../pages; git pull
	cp -r html/* ../pages/

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: clean build check upload help docs
.DEFAULT_GOAL := help
