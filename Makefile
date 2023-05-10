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

docs-si: ## Sinhalese Translation
	rm -rf docs/si
	mv docs/zh docs_zh
	mv docs/ja docs_ja
	cp -r docs docs_si
	mv docs_si docs/si
	mv docs_zh docs/zh
	mv docs_ja docs/ja
	cd labml_nn; pylit --translate si --translate_cache ../translate_cache --remove_empty_sections --title_md -t ../../../pylit/templates/nn -d ../docs/si -w *

docs-zh: ## Chinese Translation
	rm -rf docs/zh
	mv docs/si docs_si
	mv docs/ja docs_ja
	cp -r docs docs_zh
	mv docs_si docs/si
	mv docs_zh docs/zh
	mv docs_ja docs/ja
	cd labml_nn; pylit --translate zh --translate_cache ../translate_cache --remove_empty_sections --title_md -t ../../../pylit/templates/nn -d ../docs/zh -w *

docs-ja: ## Japanese Translation
	rm -rf docs/ja
	mv docs/si docs_si
	mv docs/zh docs_zh
	cp -r docs docs_ja
	mv docs_si docs/si
	mv docs_zh docs/zh
	mv docs_ja docs/ja
	cd labml_nn; pylit --translate ja --translate_cache ../translate_cache --remove_empty_sections --title_md -t ../../../pylit/templates/nn -d ../docs/ja -w *

docs: ## Render annotated HTML
	mv docs/zh docs_zh
	mv docs/si docs_si
	mv docs/ja docs_ja
	find ./docs/ -name "*.html" -type f -delete
	find ./docs/ -name "*.svg" -type f -delete
	mv docs_si docs/si
	mv docs_zh docs/zh
	mv docs_ja docs/ja
	python utils/sitemap.py
	python utils/diagrams.py
	cd labml_nn; pylit --remove_empty_sections --title_md -t ../../../pylit/templates/nn -d ../docs -w *

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: clean build check upload help docs
.DEFAULT_GOAL := help
