// The purpose of this code is to act as the first pass manipulation of incoming data
// to the automated machine learning platform such that the user does not need to
// deal with the preprocessing side only input the appropriate table and target.

\d .aml

// This is the main wrapped function used for the preprocessing of input data
// based on the type of problem being solved and the parameters supplied by the user
/* t   = initial input table
/* tgt = target data
/* typ = type of feature extraction being performed
/* p   = is a set of parameters as a dictionary or :: ('default set')
/. r   > a new table with the data preprocessed for the problem being solved
preproc:{[t;tgt;typ;p]
  prep.i.lencheck[t;tgt;typ;p];
  // For FRESH the aggregate columns need to be excluded from the preprocessing
  // steps, this ensures that encoding is not performed on the aggregate columns
  // if this is a symbol and or if this column is constant in the case of new data
  $[`fresh=typ;
    [sepdata:(p[`aggcols],())#flip t;tb:flip (cols[t]except p[`aggcols])#flip t];
    tb:t];
  show prep.i.describe tb;
  // Symbol encode the table with frequency encoding for columns with more than
  // 10 unique symbols, returning an appropriately encoded table
  tb:prep.i.symencode[tb;10;0;p;::];
  tb:.ml.dropconstant tb;
  tb:prep.i.nullencode[tb;med];
  // perform an infinity replace and rejoin the separated aggregate columns for FRESH
  $[`fresh=typ;flip sepdata,;flip]flip .ml.infreplace tb
  }
