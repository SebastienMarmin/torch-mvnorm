# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = ..

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

publish:
	rm -r -f ../_static/
	rm -r -f ../_sources/
	rm  -f ../genindex.html
	rm  -f ../index.html
	rm  -f ../objects.inv
	rm  -f ../search.html
	rm  -f ../searchindex.js
	cp -r -f ../html/* ..
	rm -r -f ../html/

	git pull
	git add ..
	git commit -m "Doc update."
	git push

