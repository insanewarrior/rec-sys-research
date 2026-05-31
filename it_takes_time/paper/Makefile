# IA-SASRec paper build targets.
#
#   make pdf    -> main.pdf via latexmk
#   make docx   -> paper.docx via pandoc (requires pandoc >= 3.0)
#   make clean  -> remove all build artefacts

.PHONY: pdf docx clean

pdf:
	latexmk -xelatex -bibtex -e '$$bibtex="biber %O %B"' -interaction=nonstopmode main.tex

docx:
	@rm -rf /tmp/docx_build && mkdir -p /tmp/docx_build/tables /tmp/docx_build/sections /tmp/docx_build/figures
	@cp main.tex docx_preamble.tex refs.bib apa-numeric.csl /tmp/docx_build/
	@cp sections/*.tex /tmp/docx_build/sections/
	@cp tables/*.tex /tmp/docx_build/tables/
	@cp figures/* /tmp/docx_build/figures/
	@sed -i '' \
		's/\\begin{table\*}/\\begin{table}/g; s/\\end{table\*}/\\end{table}/g' \
		/tmp/docx_build/tables/main_results.tex \
		/tmp/docx_build/tables/lambda_table.tex \
		/tmp/docx_build/tables/datasets.tex \
		/tmp/docx_build/sections/03_method.tex
	@cd /tmp/docx_build && pandoc docx_preamble.tex main.tex \
		--from=latex \
		--to=docx \
		--citeproc \
		--bibliography=refs.bib \
		--csl=apa-numeric.csl \
		--output=paper.docx
	@cp /tmp/docx_build/paper.docx .
	@rm -rf /tmp/docx_build

clean:
	latexmk -C
	rm -f paper.docx
