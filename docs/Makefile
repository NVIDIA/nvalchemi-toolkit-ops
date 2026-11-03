# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line or environment.
SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
# The PyData theme requires serial output; benchmark PNGs remain parallel.
DOCS_JOBS     ?= 1
BENCHMARK_PLOT_JOBS ?= auto
# Validate each bundled method collection and the complete configured case list.
BENCHMARK_ALLOW_MIXED_SNAPSHOT ?= 0
SOURCEDIR     = .
BUILDDIR      = _build
MPLCONFIGDIR  ?= $(BUILDDIR)/.matplotlib

export BENCHMARK_PLOT_JOBS MPLCONFIGDIR BENCHMARK_ALLOW_MIXED_SNAPSHOT
override SPHINXOPTS += -j $(DOCS_JOBS)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
