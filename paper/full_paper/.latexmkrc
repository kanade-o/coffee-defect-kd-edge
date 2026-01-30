#!/usr/bin/env perl
# LaTeXmk configuration for Japanese documents (platex)

$latex = 'platex -kanji=utf8 %O %S';
$bibtex = 'pbibtex %O %B';
$dvipdf = 'dvipdfmx %O -o %D %S';
$makeindex = 'mendex %O -o %D %S';
$max_repeat = 10;

# Set environment variable for TEXINPUTS to include styles directory
$ENV{'TEXINPUTS'} = './styles//:' . $ENV{'TEXINPUTS'};

# PDF generation mode (3 = LaTeX -> DVI -> PDF)
$pdf_mode = 3;

# Output directory
$out_dir = 'out';

# Continuous preview mode disabled (build finishes normally)
$preview_continuous_mode = 0;
# $pdf_previewer = 'open -a Preview';

# Clean up auxiliary files after successful build
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fdb_latexmk run tdo %R-blx.bib dvi aux log toc lof lot out fls';

# Automatically clean auxiliary files after successful build (keep only PDF)
$success_cmd = 'latexmk -c';
