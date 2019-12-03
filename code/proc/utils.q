\d .aml

// The following naming convention holds throughout this file
/* mdl = the model being applied from within the module as a symbol
/* fn  = name of a file as a string
/* fp  = file path relative to .aml.path as a string

// Utilities for proc.q

// Text files that can be parsed from within the mdldef folder
proc.i.files:`class`reg`score!("classmodels.txt";"regmodels.txt";"scoring.txt")

// Build up the model to be applied based on naming convention
/* lib = library which forms the basis for the definition
/* fnc = function name if keras or module from which model is derived for keras
/. r   > the appropriate function or projection in the case of sklearn
proc.i.mdlfunc:{[lib;fnc;mdl]
  $[any `keras`simpletransformers in raze lib;
    // retrieve keras model from the .aml namespace eg '.aml.regfitscore'
    get` sv``aml,`fitscore;
    // construct the projection used for sklearn models eg '.p.import[`sklearn.svm][`:SVC]'
    {[x;y;z].p.import[x]y}[` sv lib,fnc;hsym mdl]]}

// Update models available for use based on the number of rows in the data set
/* mdls = table defining models which are to be applied to the dataset
/* tgt  = target vector
/. r    > model table with appropriate models removed if needed and model removal highlighted
proc.i.updmodels:{[mdls;tgt]
 $[10000<count tgt;
   [-1"\nLimiting the models being applied due as the number targets exceeds 100,000";
    -1"No longer running neural nets or svms\n";
    select from mdls where(lib<>`keras),not fnc in`neural_network`svm];mdls]}


// Utilities for xvgs.q

// parse the hyperparameter flatfile
/. r  > dict mapping model name to possible hyper parameters
proc.i.paramparse:{[fn;fp]key[k]!(value@){(!).("S=;")0:x}each k:(!).("S*";"|")0:hsym`$.aml.path,fp,fn}

// The following two functions together extract the hyperparameter dictionaries
// based on the applied model
/. r   > the hyperparameters appropriate for the model being used
proc.i.edict:{[fn;fp;mdl]key[k]!value each value k:proc.i.paramparse[fn;fp]mdl}
proc.i.extractdict:proc.i.edict["hyperparams.txt";"/code/mdldef/";]


// Utilities for both scripts

// Extraction of an appropriately valued dictionary from a non complex flat file
/* sn = name mapping to appropriate text file in, as a symbol
proc.i.txtparse:{[sn;fp]{key(!).("S=;")0:x}each(!).("S*";"|")0:hsym`$path,fp,proc.i.files sn}

// Extract the appropriate ordering of output scores to allow the best model to be chosen
// these are defined in "scoring.txt"
/* scf = scoring function
/. r   > the function to order the dictionary output from cross validation search (asc/desc)
proc.i.ord:{[scf]get string first proc.i.txtparse[`score;"/code/mdldef/"]scf}
