install-pdflatex:
	sudo apt-get install latexlive
start-jupyter:
	bash -c "source activate gym && DISPLAY=:1.0 jupyter lab --ip='*'"
#	bash -c "source activate gym && jupyter notebook --ip='*'"



