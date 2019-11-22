\d .aml
separ:{"_" vs string x}

featlst:exec f from feattab:.ml.fresh.params

// Extract the table that is to be used for the application of 
// functions with(out) parameters to individual columns in a table
/* efeat = extracted features we want to build a new table from
/* cvals = columns of the original table
/. r   > a table with the function and parameter information to be applied to tabular columns
freshcolextract:{[efeat;cvals]
 fncparams:cname except coln:cvals where cvals in cname:`$separ[efeat];
 fnc:first fncparams;
 $[0<feattab[fnc]`pnum;
  [params:1_fncparams;
   ploc:where params in .ml.fresh.params[fnc]`pnames;
   pvloc:where sum("o";"w")in/:\:pv:string params ploc+1;
   paramv:enlist each"J"$pv;
   paramv[pvloc]:enlist each"F"$ssr[;"o";"."]each pv[pvloc]];
  (paramn:();paramv:())];
 `coln`f`pnum`pnames`pvals`valid!(coln;fnc;count[ploc];params ploc;paramv;1b)}

/. r   > a table with the function and parameter information to be applied to normal tabular columns
normalcolcreate:{[efeat;cvals]
 coln:cvals where cvals in cname:`$separ[efeat];
 `coln`fnc!(coln;$[count[coln]~count[cname];`norm;last cname])
 }


