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
  ord:proc.i.ord scf;
  -1"\nScores for all models, using ",string scf;
  // Score the models based on user denoted scf and ordered appropriately to find best model
  s1:ord mdls[`model]!{first avg x}each scf .''p1;
  if[`nlppretrain~p`typ;[knlp:first key[s1]where key[s1] in i.nlplist;knorm:first key[s1]where not key[s1] in i.nlplist;
    p2:proc.xv.seed[tt`xtrain;tt`ytrain;p,(`xv`prf!((`.ml.xv.kfshuff;2);`.aml.xv.fitpredictprob))]'[select from mdls where model in(knorm;knlp)];
   s1:ord s1,enlist[`Combination]!enlist avg scf'[proc.i.imax''[avg first''[p2]];first last''[p2]];
    ]];
  show s1;
  xv_tend:.z.T-xv_tstart;
  bm_tstart:.z.T;
  -1"\nBest scoring model = ",string bs:first key s1;
  bs:`Combination;
  // Extract the best model, fit on entire training set and predict/score on test set
  // for the appropriate scoring function
 preds:$[bs~`Combination;(scf[;ytst]proc.i.imax each avg probpr[;0];(probpr:proc.mdls[;xtrn;ytrn;xtst;ytst;mdls;p;0b]each knorm,knlp)[;1])
      ;(scf[;ytst]pr[0];(pr:proc.mdls[bs;xtrn;ytrn;xtst;ytst;mdls;p;1b])1)];
  if[not bs~`Combination;$[bs in i.nlplist;[tt[`xtrain]:(tt`xtrain)[;indstr:where 10h=type each first tt`xtrain];
   tt[`xtest]:(tt`xtest)[;indstr];cnms:cnms[indstr]];
   [tt[`xtrain]:(tt`xtrain)[;inorm:where not 10h=type each first tt`xtrain];tt[`xtest]:(tt`xtest)[;inorm];cnms:cnms[inorm]]]];
  -1"Score for validation predictions using best model = ",string[s2:preds 0],"\n";
  bm_tend:.z.T-bm_tstart;
  // Feature impact graph produced on holdout data if setting is appropriate
  if[p[`saveopt] in 2;$[0N!bs~`Combination;post.combfeatureimpact[bs:knorm,knlp;(bm:preds 1;mdls);value tt;cnms;scf;dt;fpath;p];
      post.featureimpact[bs;(bm:preds 1;mdls);value tt;cnms;scf;dt;fpath;p]]];
  // Outputs from run models. These are used in the generation of a pdf report
  // or are used within later sections of the pipeline.
  (s1;bs;s2;xv_tend;bm_tend;scf;bm)}

proc.mdls:{[bs;xtrn;ytrn;xtst;ytst;mdls;p;b]
     $[bs in i.keraslist;
    [data:((xtrn[;inorm];ytrn);(xtst[;inorm:where not 10h=type each first xtrn];ytst));
     funcnm:string first exec fnc from mdls where model=bs;
     if[funcnm~"multi";data[;1]:npa@'reverse flip@'./:[;((::;0);(::;1))](0,count ytst)_/:
       value .ml.i.onehot1(,/)(ytrn[;inorm];ytst)];
     kermdl:mdl[data;p`seed;`$funcnm];
     bm:fit[data;kermdl];
     s2:get[".aml.",funcnm,$[b;"predict";"predictprob"]][data;bm]];
     bs in i.nlplist;
     [data:((xtrn[;inlp];ytrn);(xtst[;inlp:where 10h=type each first xtst];ytst));
     kermdl:nlpmdl[p;bs];
     bm:nlpfit[data;kermdl];
     s2:$[b;nlppredict;nlppredictprob][data;bm]];
    [bm:(first exec minit from mdls where model=bs)[][];
     bm[`:fit][xtrn[;inorm:where not 10h=type each first xtrn];ytrn];
     s2:bm[$[b;`:predict;`:predict_proba]][xtst[;inorm]]`]
    ];(s2;bm)}
