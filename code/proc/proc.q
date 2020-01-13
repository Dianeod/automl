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
  mdls:i.kerascheck[mdls;tt;tgt];
  xv_tstart:.z.T;
  // Complete a seeded cross validation on training sets producing the predictions with associated 
  // real values. This allows the best models to be chosen based on relevant user defined metric 
  p1:proc.xv.seed[tt`xtrain;tt`ytrain;p]'[mdls];
  scf:i.scfn[p;mdls];
  ord:proc.i.ord scf;
  // Score the models based on user denoted scf and ordered appropriately to find best model
  s1:ord mdls[`model]!{first avg x}each scf .''p1;
  // If nlpretrain run Combination of best "normal" and "nlp" model
  if[(`nlp~p`typ)&1b~p`runcomb;combmdl:key[s2](where each (not kk;kk:key[s2:`SVC`LinearSVC _s1] in i.nlplist))[;0];
    p2:proc.xv.seed[tt`xtrain;tt`ytrain;p,(`xv`prf!((`.ml.xv.kfshuff;2);`.aml.xv.fitpredictprob))]'
    [select from mdls where model in combmdl];
   s1:ord s1,enlist[comb:`$"Comb_","_" sv string combmdl]!enlist avg scf'[proc.i.imax''[avg first''[p2]];first last''[p2]]];
  -1"\nScores for all models, using ",string scf;show s1;
  -1"\nBest scoring model = ",string bs:first key s1;
  xv_tend:.z.T-xv_tstart;
  bm_tstart:.z.T;
  // Extract the best model, fit on entire training set and predict/score on test set
  // for the appropriate scoring function
  preds:$[b:bs~comb;(scf[;tt`ytest]proc.i.imax each avg pr[;0];
         flip(pr:proc.i.mdls[;tt;mdls;p;0b]each combmdl)[;1 2])
        ;(scf[;tt`ytest]pr[0];(pr:proc.i.mdls[bs;tt;mdls;p;1b])1 2)];
  -1"Score for validation predictions using best model = ",string[s2:preds 0],"\n";
  bm_tend:.z.T-bm_tstart;
  // Feature impact graph produced on holdout data if setting is appropriate
  if[2=p[`saveopt];post.featureimpact[$[b;bs:combmdl;enlist bs];(bm:first preds 1;$[b;2#enlist mdls;mdls]);value tt;cnms;scf;dt;fpath;p]];
  // Outputs from run models. These are used in the generation of a pdf report
  // or are used within later sections of the pipeline.
  (s1;bs;s2;xv_tend;bm_tend;scf;last last preds;bm)}
