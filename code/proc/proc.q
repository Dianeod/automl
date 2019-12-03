\d .aml

// Run cross validated machine learning models on training data and choose the best model.
/* t     = table of features as output from preprocessing pipeline/feature extraction
/* mdls  = appropriate models from `.aml.proc.models` above
/* dt    = date and time that the run was initialized (this is used in the feature impact function) 
/* fpath = file paths for saving down information
/* tgt   = target data
/* p     = parameter dictionary passed as default or modified by user
/. r     > all relevant information about the running of the sets of models
proc.runmodels:{[data;tgt;mdls;cnms;p;dt;fpath]
  system"S ",string s:p`seed;
  // Apply train test split to keep holdout for feature impact plot and testing of vanilla best model
  tt:p[`tts][data;tgt;p`hld];
  xtrn:tt`xtrain;ytrn:tt`ytrain;xtst:tt`xtest;ytst:tt`ytest;
  mdls:i.kerascheck[mdls;tt;tgt];
  xv_tstart:.z.T;
  // Complete a seeded cross validation on training sets producing the predictions with associated 
  // real values. This allows the best models to be chosen based on relevant user defined metric 
  p1:proc.xv.seed[tt`xtrain;tt`ytrain;p]'[mdls];
  
  scf:i.scfn[p;mdls];
  if[not typchk:p[`typ]~`nlpclass;
   ord:proc.i.ord scf;
   -1"\nScores for all models, using ",string scf;
   // Score the models based on user denoted scf and ordered appropriately to find best model
   show s1:ord mdls[`model]!{first avg x}each scf .''p1];
  xv_tend:.z.T-xv_tstart;
  -1"\nBest scoring model = ",string bs:$[typchk;first mdls`model;first key s1];
  // Extract the best model, fit on entire training set and predict/score on test set
  // for the appropriate scoring function
  bm_tstart:.z.T;
  $[typchk;[modl:first exec minit from mdls;
    bm:modl[((tt`xtrain;tt`ytrain);(tt`xtrain;tt`ytest));p];
    s2:scf[;ytst:tt`ytest]first bm[`:predict][raze each xtst:tt`xtest]`];
   [bm:(first exec minit from mdls where model=bs)[][];
    bm_tstart:.z.T;
    bm[`:fit][tt`xtrain;tt`ytrain];
    s2:scf[;ytst:tt`ytest]bm[`:predict][xtst:tt`xtest]`]];
  -1"Score for validation predictions using best model = ",string[s2],"\n";
  bm_tend:.z.T-bm_tstart;
  // Feature impact graph produced on holdout data if setting is appropriate
  if[2=p[`saveopt];post.featureimpact[bs;(bm;mdls);value tt;cnms;scf;dt;fpath;p]];
  // Outputs from run models. These are used in the generation of a pdf report
  // or are used within later sections of the pipeline.
  (s1;bs;s2;xv_tend;bm_tend;scf;bm)}
