\d .automl

// The following is a naming convention used in this file
/* d = data as a mixed list containing training and testing data ((xtrn;ytrn);(xtst;ytst))
/* m = model object being passed through the system (compiled/fitted)
/* dict = dictionary of parameters
/* mdl = dictionary of model being used
/. r > the predicted values for a given model as applied to input data

i.nlplist:`Bert`RoBERTa`DistilBERT`ALBERT`XLM`XLNet;

nlpfitscore:{[d;dict;mdl]		
  m:nlpmdl[dict;mdl];		
  m:nlpfit[d;m];		
  nlppredict[d;m]}

nlpdict:(!). flip(
  (`bert;         "bert-base-uncased");
  (`roberta;      "roberta-base");
  (`xlnet;        "xlnet-base-cased");
  (`xlm;          "xlm-mlm-en-2048");
  (`distilbert;   "distilbert-base-uncased-distilled-squad");
  (`albert;     "albert-base-v1");
  (`camembert;    "camembert-base"))


nlpmdl:{[dict;mdls] 
  args:`overwrite_output_dir`use_multiprocessing`output_dir`cache_dir`silent`reprocess_input_data`regression`tensorboard_dir!
   (1b;0b;pth,"/models/",string[mdln:lower$[type[mdls]in neg[11h];mdls;first mdls`model]];
   pth,"/cache_dir";1b;0b;regmd:`reg~dict[`ptyp];(pth:path,"/",dict[`spath]),"/runs");
  args,:dict`args;
  system["S ",string[dict`seed]];
  nps[dict`seed];
  tfs[dict`seed];
  trmseed[dict`seed];
  modeln:$[`best_model in key dict;pth,"/models/",string[mdln];nlpdict[mdln]];
  pydict:`model_type`model_name`use_cuda`num_labels`args!
    (mdln;modeln;0b;$[regmd;1;dict`tgtnum];args);
  m:nlpclass[pykwargs pydict];m}

/. r > the fit simpletransformers model
nlpfit:{[d;m]m[`:train_model][pdD((d[0]0),'enlist each d[0]1)];m}

// Prediction functions for each of the keras models
/* d = Data from which prediction is to be made 
/*     formatting based on master workflow ((0n;0n);(xtst;0n))
/. r > predicted values
nlppredict  :{[d;m]first m[`:predict][$[0h~type d[1]0;raze;] d[1]0]`}

nps:.p.import[`numpy.random][`:seed];
pdD:.p.import[`pandas]`:DataFrame;
nlpclass:.p.import[`simpletransformers.classification]`:ClassificationModel;
trmseed:.p.import[`torch][`:manual_seed];
tf:.p.import[`tensorflow];tfs:tf$[2>"I"$first tf[`:__version__]`;[`:set_random_seed];[`:random.set_seed]]

/ allow multiprocess
.ml.loadfile`:util/mproc.q
if[0>system"s";.ml.mproc.init[abs system"s"]("system[\"l automl/automl.q\"]";".automl.loadfile`:init.q")];
