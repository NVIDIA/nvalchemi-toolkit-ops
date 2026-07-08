# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line or environment.
SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
# The PyData theme requires serial output; benchmark PNGs remain parallel.
DOCS_JOBS     ?= 1
BENCHMARK_PLOT_JOBS ?= auto
SOURCEDIR     = .
BUILDDIR      = _build
MPLCONFIGDIR  ?= $(BUILDDIR)/.matplotlib

export BENCHMARK_PLOT_JOBS MPLCONFIGDIR
override SPHINXOPTS += -j $(DOCS_JOBS)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
