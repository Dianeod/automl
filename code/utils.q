\d .aml

// Utilities for run.q

/  Function takes in a string which is the name of a parameter flatfile
/  Returns the parameter dictionary
i.getdict:{
 d:i.paramparse[x;"/code/mdl_def/"];
 idx:(k except`scf;k except`xv`gs`scf;$[`xv in k;`xv;()],$[`gs in k;`gs;()];$[`scf in k:key d;`scf;()]);
 fnc:(key;{get string first x};{(x 0;get string x 1)};{key[x]!`$value x});
 if[sgl:1=count d;d:(enlist[`]!enlist""),d];
 d:{$[0<count y;@[x;y;z];x]}/[d;idx;fnc];
 if[sgl;d:1_d];
 d}
  
/  This function sets or updates the default parameter dictionary as appropriate
/* x   = data as table
/* p   = dictionary of parameters (type of feature extract dependant)
/* typ = type of feature extraction (FRESH/normal/tseries ...)
i.updparam:{[x;p;typ]
 dict:
  / FRESH
  $[typ=`fresh;
      {d:`aggcols`params`xv`gs`prf`scf`seed`saveopt`hld`tts`sz!
         ({first cols x};.ml.fresh.params;(`kfshuff;5);(`kfshuff;5);
         xv.fitpredict;`class`reg!(`.ml.accuracy;`.ml.mse);
         42;2;0.2;.ml.traintestsplit;0.2);	   
       d:$[(ty:type y)in 10 -11 99h;
	      [if[10h~ty;y:.aml.i.getdict y];
		   if[-11h~ty;y:.aml.i.getdict$[":"~first y;1_;]y:string y];
		   $[min key[y]in key d;d,y;
			 '`$"You can only pass appropriate keys to fresh"]];
		  y~(::);d;
		  '`$"p must be passed the identity `(::)`, a filepath to a parameter flatfile or a dictionary with appropriate key/value pairs"];
	   d[`aggcols]:$[100h~typagg:type d`aggcols;d[`aggcols]x;11h~abs typagg;d`aggcols;'`$"aggcols must be passed function or list of columns"];
	   d,enlist[`tf]!enlist 0~checkimport[]}[x;p];
  / NORMAL
    typ=`normal;
      {d:`xv`gs`prf`scf`seed`saveopt`hld`tts`sz!
         ((`kfshuff;5);(`kfshuff;5);xv.fitpredict;
         `class`reg!(`.ml.accuracy;`.ml.mse);
         42;2;0.2;.ml.traintestsplit;0.2);
       d:$[(ty:type y)in 10 -11 99h;
	      [if[10h~ty;y:.aml.i.getdict y];
		   if[-11h~ty;y:.aml.i.getdict$[":"~first y;1_;]y:string y];
		   $[min key[y]in key d;d,y;
			 '`$"You can only pass appropriate keys to normal"]];
		  y~(::);d;
		  '`$"p must be passed the identity `(::)`, a filepath to a parameter flatfile or a dictionary with appropriate key/value pairs"];
	   d,enlist[`tf]!enlist 0~checkimport[]}[x;p];
  / TIMESERIES
    typ=`tseries;
      '`$"This will need to be added once the time-series recipe is in place";
  / ERROR
    '`$"Incorrect input type"]}

/  apply scoring function to precitions from model
/* x = x-test; y = y-test; z = model; r = scoring function
i.scorepred:{[x;y;z;r] r[;y]z[`:predict][x]`}

/  save down the best model
/* x = date-time of model start (dict)
/* y = best model name (`symbol)
/* z = best model object (embedPy)
/* r = all applied models (table)
i.savemdl:{[x;y;z;r]
 folder_name:path,"/",mo:ssr["Outputs/",string[x`stdate],"/Run_",string[x`sttime],"/Models/";":";"."];
 save_path: system"mkdir -p ",folder_name;
 joblib:.p.import[`joblib];
 $[(`sklearn=?[r;enlist(=;`model;y,());();`lib])0;
    (joblib[`:dump][z;folder_name,"/",string[y]];-1"Saving down ",string[y]," model to ",mo);
   (`keras=?[r;enlist(=;`model;y,());();`lib])0;
    (bm[`:save][folder_name,"/",string[y],".h5"];-1"Saving down ",string[y]," model to ",mo);
   -1"Saving of non keras/sklearn models types is not currently supported"];
 }

\d .
\d .ml

i.infrep:{
 t:i.inftyp first string y;
 {[n;x;y;z]@[x;i;:;z@[x;i:where x=y;:;n]]}[t 0]/[x;t 1 2;(min;max)]}
i.inftyp:("5";"8";"9";"6";"7";"12";"16";"17";"18")!
 (0N -32767 32767;0N -0w 0w;0n -0w 0w),6#enlist 0N -0W 0W

infreplace:{
 $[98=t:type x;
   [m:type each dt:k!x k:.ml.i.fndcols[x;"hijefpnuv"];flip flip[x]^i.infrep'[dt;m]];
   0=t;
   [m:type each dt:x r:where all each string[type each x]in key i.inftyp;(x til[count x]except r),i.infrep'[dt;m]];
   98=type kx:key x;
   [m:type each dt:k!x k:.ml.i.fndcols[x:value x;"hijefpnuv"];cols[kx]xkey flip flip[kx],flip[x]^i.infrep'[dt;m]];
   [m:type each dt:k!x k:.ml.i.fndcols[x:flip x;"hijefpnuv"];flip[x]^i.infrep'[dt;m]]]}

