# DART paper build
MAIN = main
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

.PHONY: all clean figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib figures/*.pdf
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

figures:
	python3 scripts/plot_architecture.py
	python3 scripts/plot_error_analysis.py
	python3 scripts/plot_periodic_table.py
	python3 scripts/plot_ablation_waterfall.py
	python3 scripts/plot_loho.py
	python3 scripts/plot_uq_calibration.py

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,fls,fdb_latexmk,synctex.gz}
	rm -f sections/*.aux
