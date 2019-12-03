\d .aml

// The following aspects of the naming parameter naming are used throughout this file
/* t   = data as table
/* p   = dictionary of parameters (type of feature extract dependant)
/* typ = type of feature extraction (FRESH/normal/tseries ...)
/* tgt  = target vector
/* mdls = table denoting all the models with associated information used in this repository


// Utilities for run.q

//  This function sets or updates the default parameter dictionary as appropriate
i.updparam:{[t;p;typ]
  dict:
    $[typ=`fresh;
      {[t;p]d:i.freshdefault[];	   
       d:$[(ty:type p)in 10 -11 99h;
	   [if[10h~ty;p:.aml.i.getdict p];
	    if[-11h~ty;p:.aml.i.getdict$[":"~first p;1_;]p:string p];
	    $[min key[p]in key d;d,p;'`$"You can only pass appropriate keys to fresh"]];
           p~(::);d;
             '`$"p must be passed the identity `(::)`, a filepath to a parameter flatfile",
                " or a dictionary with appropriate key/value pairs"];
	   d[`aggcols]:$[100h~typagg:type d`aggcols;
                         d[`aggcols]t;
                         11h~abs typagg;d`aggcols;
                         '`$"aggcols must be passed function or list of columns"];
	   d,enlist[`tf]!enlist 1~checkimport[]}[t;p];
      typ=`normal;
      {[t;p]d:i.normaldefault[];
       d:$[(ty:type p)in 10 -11 99h;
	   [if[10h~ty;p:.aml.i.getdict p];
            if[-11h~ty;p:.aml.i.getdict$[":"~first p;1_;]p:string p];
            $[min key[p]in key d;d,p;
	      '`$"You can only pass appropriate keys to normal"]];
           p~(::);d;
	   '`$"p must be passed the identity `(::)`, a filepath to a parameter flatfile",
              " or a dictionary with appropriate key/value pairs"];
	   d,enlist[`tf]!enlist 1~checkimport[]}[t;p];
      typ=`nlpclass;
       {[t;p]d:i.nlpclassdefault[];
       d:$[(ty:type p)in 10 -11 99h;
           [if[10h~ty;p:.aml.i.getdict p];
            if[-11h~ty;p:.aml.i.getdict$[":"~first p;1_;]p:string p];
            $[min key[p]in key d;d,p;
              '`$"You can only pass appropriate keys to normal"]];
           p~(::);d;
           '`$"p must be passed the identity `(::)`, a filepath to a parameter flatfile",
              " or a dictionary with appropriate key/value pairs"];
           d,enlist[`tf]!enlist 1~checkimport[]}[t;p];
      typ=`tseries;
      '`$"This will need to be added once the time-series recipe is in place";
    '`$"Incorrect input type"]}

//  Function takes in a string which is the name of a parameter flatfile
/* nm = name of the file from which the dictionary is being extracted
/. r  > the dictionary as defined in a float file in mdldef
i.getdict:{[nm]
  d:proc.i.paramparse[nm;"/code/mdldef/"];
  idx:(k except`scf;
    k except`xv`gs`scf`seed;
    $[`xv in k;`xv;()],$[`gs in k;`gs;()];
    $[`scf in k;`scf;()];
    $[`seed in k:key d;`seed;()]);
  fnc:(key;
       {get string first x};{(x 0;get string x 1)};
       {key[x]!`$value x};{$[`rand_val~first x;first x;get string first x]});
  // Addition of empty dictionary entry needed as parsing 
  // of file behaves oddly if only a single entry is given to the system
  if[sgl:1=count d;d:(enlist[`]!enlist""),d];
  d:{$[0<count y;@[x;y;z];x]}/[d;idx;fnc];
  if[sgl;d:1_d];
  d}

// Default parameters used in the population of parameters at the start of a run
// or in the creation of a new initialisation parameter flat file
/* Neither of these function take a parameter as input
/. r > default dictionaries which will be used by the automl
i.freshdefault:{`aggcols`params`xv`gs`prf`scf`seed`saveopt`hld`tts`sz!
  ({first cols x};`.ml.fresh.params;(`.ml.xv.kfshuff;5);(`.ml.gs.kfshuff;5);`.aml.xv.fitpredict;
   `class`reg!(`.ml.accuracy;`.ml.mse);`rand_val;2;0.2;`.ml.ttsnonshuff;0.2)}
i.normaldefault:{`xv`gs`prf`scf`seed`saveopt`hld`tts`sz!
  ((`.ml.xv.kfshuff;5);(`.ml.gs.kfshuff;5);`.aml.xv.fitpredict;`class`reg!(`.ml.accuracy;`.ml.mse);
   `rand_val;2;0.2;`.ml.traintestsplit;0.2)}
i.nlpclassdefault:{`model_type`model_name`args`xv`gs`prf`scf`seed`saveopt`hld`tts`sz!
  (`bert;"bert-base-uncased";();(`.ml.xv.kfshuff;5);(`.ml.gs.kfshuff;5);`.aml.xv.fitpredict;`class`reg!(`.ml.accuracy;`.ml.mse);
   `rand_val;2;0.2;`.ml.traintestsplit;0.2)}

// Apply an appropriate scoring function to predictions from a model
/* xtst = test data
/* ytst = test target
/* mdl  = fitted embedPy model object
/* scf  = scoring function which determines best model
/. r    > score for the model based on the predictions on test data
i.scorepred:{[xtst;ytst;mdl;scf]scf[;ytst]mdl[`:predict][xtst]`}

/  save down the best model
/* dt = date-time of model start (dict)
/* bmn = best model name (`symbol)
/* bmo = best model object (embedPy)
/* r = all applied models (table)
i.savemdl:{[bmn;bmo;mdls;nms]
  fname:nms[0]`models;mo:nms[1]`models;
  system"mkdir -p ",fname;
  joblib:.p.import[`joblib];
  $[(`sklearn=?[mdls;enlist(=;`model;bmn,());();`lib])0;
      (joblib[`:dump][bmo;fname,"/",string[bmn]];-1"Saving down ",string[bmn]," model to ",mo);
    (`keras=?[mdls;enlist(=;`model;bmn,());();`lib])0;
      (bmo[`:save][fname,"/",string[bmn],".h5"];-1"Saving down ",string[bmn]," model to ",mo);
    (`simpletransformers=?[mdls;enlist(=;`model;bmn,());();`lib])0;
      -1"Saving down ",string[bmn]," model to ",mo;
   -1"Saving of non keras/sklearn models types is not currently supported"];
 }

// Table of models appropriate for the problem type being solved
/* ptyp = problem type as a symbol, either `class or `reg
/. r   > table with all information needed for appropriate models to be applied to data
i.models:{[ptyp;tgt;p]
  if[not ptyp in key proc.i.files;'`$"text file not found"];
  d:proc.i.txtparse[ptyp;"/code/mdldef/"];
  if[1b~p`tf;
    d:l!d l:key[d]where not `keras=first each value d];
  m:flip`model`lib`fnc`seed`typ!flip key[d],'value d;
  if[ptyp=`class;
    // For classification tasks remove inappropriate classification models
    m:$[`nlpclass~p`typ;[$[1~count tgt[0];
        select from m where model=`NLPmultiClass;
        select from m where model=`NLPmultiLabel]];
        2<count distinct tgt;
        delete from m where typ=`binary;
        delete from m where model=`MultiKeras]];
  // Add a column with appropriate initialized models for each row
  m:update minit:.aml.proc.i.mdlfunc .'flip(lib;fnc;model)from m;
  // Threshold models used based on unique target values
  i.updmodels[m;tgt]}

// Update models available for use based on the number of rows in the data set
/. r    > model table with appropriate models removed if needed and model removal highlighted
i.updmodels:{[mdls;tgt]
 $[100000<count tgt;
   [-1"\nLimiting the models being applied due to number targets>100,000";
    -1"No longer running neural nets or svms\n";
    select from mdls where(lib<>`keras),not fnc in`neural_network`svm];mdls]}

// These are a list of models which are deterministic and thus which do not need to be grid-searched 
// at present this should include the Keras models as a sufficient tuning method
// has yet to be implemented
i.excludelist:`GaussianNB`LinearRegression`RegKeras`MultiKeras`BinKeras;

// Dictionary with mappings for console printing to reduce clutter in .aml.runexample
i.runout:`col`pre`sig`slct`tot`ex`gs`sco`save!
 ("\nThe following is a breakdown of information for each of the relevant columns in the dataset\n";
  "\nData preprocessing complete, starting feature creation";
  "\nFeature creation and significance testing complete";
  "Starting initial model selection - allow ample time for large datasets";
  "\nTotal features being passed to the models = ";
  "Continuing to final model fitting on holdout set";
  "Continuing to grid-search and final model fitting on holdout set";
  "\nBest model fitting now complete - final score on test set = ";
  "Saving down procedure report to ")


// Save down the metadata dictionary as a binary file which can be retrieved by a user or
// is to be used in running of the models on new data
/* d     = dictionary of parameters to be saved
/* dt    = dictionary with the date and time that a run was started, required for naming of save path 
/* fpath = dictionary of file paths for saving
/. r     > the location that the metadata was saved to
i.savemeta:{[d;dt;fpath]
  `:metadata set d;
  // move the metadata information to the appropriate location based on OS
  $[first[string .z.o]in "lm";
    system"mv metadata ",;
    system"move metadata ",]fpath[0]`config;
  -1"Saving down model parameters to ",fpath[1]`config;}

// Retrieve the metadata information from a specified path
/* fp = full file path denoting the location of the metadata to be retrieved
/. r  > returns the parameter dictionary
i.getmeta:{[fp]
  fp:`$":",fp;
  $[()~key fp;'`$"metadata file doesn't exist";get fp]
  }


// Apply feature creation and encoding procedures for 'normal' on new data
/. r > table with feature creation and encodings applied appropriately
i.normalproc:{[t;p]
  t:i.rmvunder t;
  prep.i.autotype[t;p`typ;p];
  // Symbol encoding completed based on encoding applied in a previous 'run'
  t:prep.i.symencode[t;10;0;p;p`symencode];
  t:prep.i.nullencode[t;med];
  t:.ml.infreplace[t];
  t:(`$ssr[;"_";""]each string cols t)xcol t;
  // Create the table needed to extract features on a column/function basis
  normtab:i.normecols[;cols t]each p[`features];
  // Names of all the columns which truncated singular value decomposition is applied
  trunccol:exec coln from normtab where fnc=`trsvd;
  t:prep.i.truncsvd[t;trunccol;::];
  bulkt:select from normtab where not fnc in `norm`trsvd;
  if[0<count[bulkt];
    // new table denoting how new functions are to be applied to data
    ntab:0!select coln by fnc from select fnc by coln from bulkt;
    // perform transfomation as appropriate for each required bulk modification
    t:(uj/){prep.i.bulktransform[y;x`coln;x`fnc;0b]}[;t]each ntab];
  flip value flip p[`features]#t}
  

// Apply feature creation and encoding procedures for FRESH on new data
/. r > table with feature creation and encodings applied appropriately
i.freshproc:{[t;p] 
  t:prep.i.autotype[t;p`typ;p];
  agg:p`aggcols;
  // Apply symbol encoding based on a previous run of automl
  t:prep.i.symencode[t;10;0;p;p`symencode];
  t:i.rmvunder t;
  cols2use:k where not (k:cols t)in agg;
  // Create table denoting all functions to be applied to each column
  appfncs:`coln xgroup distinct i.freshecols[;cols t]each p`features;
  // Modified fresh feature creation for column by column extraction
  fproc:{[x;y;z;r].ml.fresh.createfeatures[x;y;z;`f xkey flip r]};
  // Feature creation applied on appropriate columns
  t:prep.i.nullencode[0!(uj/)fproc[t;agg]'[(0!appfncs)`coln;value appfncs];med];
  t:value agg xkey p[`features]xcols .ml.infreplace t;
  // It is not guaranteed that new feature creation will produce the all requisite features 
  // if this is not the case dummy features are added to the data, column reordering may be
  // required in some circumstances.
  if[not all ftc:p[`features]in cols t;
    newcols:p[`features]where not ftc;
    t:p[`features]xcols flip flip[t],newcols!((count newcols;count t)#0f),()];
  flip value flip p[`features]#"f"$0^t}


// Extract the table that is to be used for the application of 
// functions with(out) parameters to individual columns in a table
/* efeat = extracted features we want to build a new table from
/* cvals = columns of the original table
/. r   > a table with the function and parameter information to be applied to tabular columns
i.freshecols:{[efeat;cvals]
  // Separate the column names for use in extracting features appropriately
  sepcols:`$i.separ[efeat];
  // Extract the required original table column name
  coln:cvals where cvals in sepcols;
  // Extract function and hyper parameters from the extracted feature names
  fncparams:sepcols except coln;
  // Extract function name
  fnc:first fncparams;
  $[0<.ml.fresh.params[fnc]`pnum;
    [params:1_fncparams;
     // For a given function extract the location of parameter names
     ploc:where params in .ml.fresh.params[fnc]`pnames;
     // The string representation of hyperparameter values
     pv:string params ploc+1;
     // Find all locations that need to be converted to floats
     pvloc:where sum("o";"w")in/:\:pv;
     paramv:enlist each"J"$pv;
     // Extract the floating point representation of relevant parameters
     paramv[pvloc]:enlist each"F"$ssr[;"o";"."]each pv[pvloc]];
     (paramn:();paramv:())];
  `coln`f`pnum`pnames`pvals`valid!(coln;fnc;count[ploc];params ploc;paramv;1b)}

/. r   > a table with the function and parameter information to be applied to normal tabular columns
i.normecols:{[efeat;cvals]
  coln:cvals where cvals in cname:`$i.separ[efeat];
  `coln`fnc!(coln;$[count[coln]~count[cname];`norm;last cname])
  }

i.separ:{"_" vs string x}

// Create the folders that are required for the saving of the config,models, images and reports
/* dt  = date and time dictionary denoting the start of a run
/* svo = save option defined by the user, this can only be 1/2 in this case
/. r   > the file paths in its full format or truncated for use in outputs to terminal
i.pathconstruct:{[dt;svo]
  names:`config`models;
  if[svo=2;names:names,`images`report]
  pname:{"/",ssr["outputs/",string[x`stdate],"/run_",string[x`sttime],"/",y,"/";":";"."]};
  paths:path,/:pname[dt]each string names;
  paths:i.ssrwin[paths];
  {[fnm]system"mkdir",$[.z.o like "w*";" ";" -p "],fnm}each paths;
  (names!paths;names!{count[path]_x}each paths)
  }


// Util functions used in multiple util files

// Error flag if test set is not appropriate for multiKeras model
/. r    > the models table with the MultiKeras model removed
i.errtgt:{[mdls]
  -1 "\n Test set does not contain examples of each class. Removed MultiKeras from models";
  delete from mdls where model=`MultiKeras}

// Extract the scoring function to be applied for model selection
/. r    > the scoring function appropriate to the problem being solved
i.scfn:{[p;mdls]p[`scf]$[`reg in distinct mdls`typ;`reg;`class]}

// Check if MultiKeras model is to be applied and each target exists in both training and testing sets
/* tts  = train-test split dataset
/. r    > table with multi-class keras model removed if it is not to be applied
i.kerascheck:{[mdls;tts;tgt]
  mkcheck :(`MultiKeras in mdls`model);
  tgtcheck:(count distinct tgt)>min{count distinct x}each tts`ytrain`ytest;
  $[mkcheck&tgtcheck;i.errtgt;]mdls}

// Used throughout the library to convert windows/mac file names to windows equivalent
/* path = the linux 'like' path
/. r    > the path modified to be suitable for windows systems
i.ssrwin:{[path]$[.z.o like "w*";ssr[path;"/";"\\"];path]}

// Remove underscores from column names, this is needed to ensure that running on new 
// data can make use of functions that speed up execution of feature extraction
/. r > table with columns renamed appropriately
i.rmvunder:{[t](`$ssr[;"_";""]each string cols t) xcol t}
 
i.precision_microavg:{.ml.precision[raze x;raze y;1b]}
i.sensitivity_microavg:{.ml.sensitivity[raze x;raze y;1b]}

i.macroavg:{(sum z[;;1b]'[flip x;flip y])%count first x}
i.precision_macroavg:{i.macroavg[x;y;.ml.precision]}
i.sensitivity_macroavg:{i.macroavg[x;y;.ml.sensitivity]}
