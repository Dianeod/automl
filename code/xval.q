\d .ml

xval.i.shuffle:{neg[n]?n:count x}

xval.i.splitidx:{[k;x](k;0N)#til count x}
xval.i.shuffidx:{[k;x](k;0N)#xval.i.shuffle x}
xval.i.stratidx:{[k;x]r@'xval.i.shuffle each r:(,'/)(k;0N)#/:value n@'xval.i.shuffle each n:group x}
xval.i.idx:{[k](0;k-1)_/:rotate[-1]\[til k]}
xval.i.idxgen:{[f;g;k;n;x;y]raze n#enlist{{raze@''y}[;x]}each flip@'((x;y)@/:\:f[k;y])@\:/:g k}

xval.i.apply:{[idx;k;n;x;y;a;f]
 {[a;f;b;d]f[$[b;a[];a]]d[]}[$[b;xval.i.pickledump a;a];f;b:105h~type a]peach idx[k;n;x;y]}
xval.kfsplit:xval.i.apply xval.i.idxgen . xval.i`splitidx`idx
xval.kfshuff:xval.i.apply xval.i.idxgen . xval.i`shuffidx`idx
xval.kfstrat:xval.i.apply xval.i.idxgen . xval.i`stratidx`idx
xval.tsroll :xval.i.apply xval.i.idxgen[xval.i.splitidx]{[k]enlist@''0 1+/:til k-1}
xval.tschain:xval.i.apply xval.i.idxgen[xval.i.splitidx]{[k]flip(til each j;enlist@'j:1+til k-1)}
xval.mcsplit:xval.i.apply{[p;n;x;y]
 n#{[p;x;y;z](x;y)@\:/:(0,floor count[y]*1-p)_{neg[n]?n:count x}y}[p;x;y]}

xval.gridsearch:{[xv;x;y;a;f;pd]p!xv[x;y;a]each f@/:pykwargs@/:p:key[pd]!/:1_'(::)cross/value pd}
xval.gridsearchfit:{[xv;x;y;a;f;pd;pc]
  i:(0,floor count[y]*1-pc)_xval.i.shuffle y;
  pr:first key desc avg each r:xval.gridsearch[xv;x i 0;y i 0;a;f;pd];
  (pr;f[pykwargs pr;a](x;y)@\:/:i)}

xval.fitscore:{[p;a;d]$[105h~type a;.[.[a[p]`:fit;d 0]`:score;d 1]`;a[d;p]]}

/ allow multiprocess
loadfile`:util/mproc.q
loadfile`:util/pickle.q
if[0>system"s";mproc.init[abs system"s"]enlist".ml.loadfile`:util/pickle.q"];
xval.i.pickledump:{$[(0>system"s")&.p.i.isw x;{.ml.pickleload y}[;pickledump x];{y}[;x]]}
