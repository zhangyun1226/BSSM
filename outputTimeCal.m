clear;
close all;
load spation2.mat inputWeight allPSPTrain allOsc outputColumn
initParameter;
load data22.mat
sysinput = 1000;
sampleNum = length(inputLablesCell);
outputTime = nt*ones(sampleNum,activeNum);
targetColumn = zeros(sampleNum,activeNum);
classNum = sampleNum;
ca3value = 0.1*Vnorm;
%% 计算膜电压和柱的输出
AllV = zeros(nt*classNum,columnNum);         %存储所有柱在一个周期的膜电压
CA3IV = zeros(nt*classNum,columnNum);
refV = zeros(nt*classNum,columnNum);
AllTout = zeros(columnNum,1);
multiOut = {};
multiColumn = [];
multiNum = 1;
for word = 1:sampleNum
        startTime = (word-1)*nt;
        input = startTime+inputCell{word};                     %取出每个样本
        intTime = Inf*ones(columnNum,1);                       %抑制神经元的输出
        toutCA3E = Inf*ones(columnNum,1);                     %柱在该周期的输出
        toutCa3Last = -Inf*ones(columnNum,1);                  %柱在该周期最近一次的输出
        fireNeurons = zeros(columnNum,1);                      %哪些柱点火
        for t = startTime+(1:nt)*dt                         %算该神经元的点火时间，遍历0到Tmax
            tIdx = t/dt;
            %% 计算前馈输入的产生的PSP以及前馈电压
            currentT = tIdx-startTime;
            PSPInput = allPSPTrain(word,:,currentT)';
            inputV = inputWeight'*PSPInput;
            %% 不应期
            ref = -exp((toutCa3Last-t)/Tau_a);
            %% 抑制PSP
            tempInh = t - intTime;
            tempInh(tempInh<=0) = inf;
            CA3I = -ca3value*sum(exp(-tempInh/Tau_s),2);
            %% 计算柱的膜电压
            CA3EV = inputV + CA3I + ref+allOsc(:,currentT);
            %% 记录
            AllV(tIdx,:) = CA3EV;              
            CA3IV(tIdx,:) = CA3I;
            refV(tIdx,:) = ref;
            %% 存在超过阈值的神经元
            neuron = find(CA3EV >=Theta);      %膜电压是否大于阈值点火
            if(length(neuron))                 %若存在神经元点火
                tempOut = toutCA3E(neuron);
                if all(tempOut == Inf)
                    toutCA3E(neuron) = t;
                else
                    tempIdx = find(tempOut~=Inf);
                    toutCA3E(neuron(setdiff(1:length(tempOut),tempIdx)),end) = t;
                    toutCA3E(:,end+1) = Inf;
                    toutCA3E(neuron(tempIdx),end) = t;
                end
                toutCa3Last(neuron) = t;
                AllTout(neuron,end+1) = t;
                fireNeurons(neuron) = fireNeurons(neuron)+1;
                currentOut = t*ones(columnNum,1);
                currentOut(neuron) = Inf;
                intTime(:,end+1) = currentOut;
            end
        end
    currentcolumn = find(fireNeurons>0);
    if length(currentcolumn)~=10
        word
    else
         targetColumn(word,:) = currentcolumn';
         currentTime = toutCa3Last(currentcolumn)-startTime;
         if size(toutCA3E,2)>1
             multiColumn(end+1) = word;
             multiOut{multiNum} = toutCA3E(currentcolumn,:)-startTime;
             multiNum = multiNum+1;
         end
         outputTime(word,:) = currentTime;
    end
   
end
% str = 'spation';
% str = [str,columnNum];
% str = [str,'.mat'];
% save(str,'inputWeight','allPSPTrain','allOsc','outputColumn','outputTime'); 
