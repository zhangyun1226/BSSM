parameters = 0;
preall = 0;
postall = 0;
for col = 1:425
    struc = column(col).neuron;
    len = length(struc);
    preNum = [];
    postNum = [];
    for m = 1:len
     preNum(m)= numel(struc(m).precessor);
     postNum(m) = numel(struc(m).successor);
    end
    preall = preall + sum(preNum);
    postall = postall + sum(postNum);
end
preall
postall