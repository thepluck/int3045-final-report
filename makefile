export TEXINPUTS=.:latex/:
LATEX=pdflatex
FLAGS=-halt-on-error -shell-escape
BUILD_FOLDER=build/
PDF_NAME=int3405-final-report.pdf
OUTPUT=$(BUILD_FOLDER)/$(PDF_NAME)
SHELL := /bin/bash

all: report

install-texlive-ubuntu:
	sudo apt update
	sudo apt install \
		ghostscript \
		texlive \
		texlive-fonts-recommended \
		texlive-latex-extra \
		texlive-lang-other \
		texlive-plain-generic

clean:
	rm src/*.log
	rm src/*.out
	rm src/*.toc
	rm src/*.aux

$(BUILD_FOLDER):
	mkdir -p $(BUILD_FOLDER)
	
report: render-articles | $(BUILD_FOLDER)
	make render-pdf
	
render-articles:
	export PYTHONPATH="${PYTHONPATH}:./scripts/"; \
	for article in ./articles/*.md; do \
		echo Processing $$article; \
		marko -e marko_latex_extension -o src/$${article//.md/.latex} $$article; \
	done

render-pdf: $(BUILD_FOLDER)
	cd src; \
	$(LATEX) $(FLAGS) report.latex; \
	cd ..; \
	cp src/report.pdf $(OUTPUT)
