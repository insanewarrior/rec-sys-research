# IA-SASRec paper build targets.
#
#   make pdf    -> main.pdf via latexmk
#   make docx   -> paper.docx via pandoc (requires pandoc >= 3.0)
#   make clean  -> remove all build artefacts

.PHONY: pdf docx clean

pdf:
	latexmk -xelatex -bibtex -e '$$bibtex="biber %O %B"' -interaction=nonstopmode main.tex

docx:
	pandoc main.tex \
	    --from=latex \
	    --to=docx \
	    --citeproc \
	    --bibliography=refs.bib \
	    --output=paper.docx

clean:
	latexmk -C
	rm -f paper.docx
