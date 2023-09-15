clear;
close;
%% 空间层-编码层
%% 参数设置
load('data1000.mat');                            %加载相应的数据
trialNum = 1;

AllEpoch = zeros(1:trialNum);
for trial = 1:trialNum
    learnRatePos = 0.0015;
    learnRateNeg = 0.001;
    columnNum = 512;
    activeNum = 10;
    maxEpoch = 200;                               %训练的epoch数量
    sysinput = 1000;
    inputWeight = normrnd(0.01,0.01,sysinput,columnNum); %权重初始化方式：高斯随机
    weight_start = inputWeight;                          %保存初始权重
    %% STDP para
    maxWeight = 1/activeNum;          %% 最大权重
    minWeight = -1/activeNum;         %% 最小权重
    tau=25;               %% STDP的参数
    %% LIF
    Theta = 1;
    Tau_m = 10;
    Tau_s = 2.5;
    Tau_a = 5;
    Eta = Tau_m/Tau_s;
    Vnorm = (Eta^(Eta/(Eta-1)))/(Eta-1);
    
    T = 100;
    dt = 1;
    nt = round(T/dt);
    ca3value=0.1*Vnorm;
    blackTheta = 3;
    blackRedius = 10;
    %% theta  oscillation
    fDG = 10;
    through = 1;
    period = T;
    phasePre = 2*pi*rand(columnNum,1);
    inputCA3 = 0.15*[cos(2*pi*fDG*(1:nt)*dt*1e-3+phasePre)+1];
    allOsc = inputCA3;  %行是每个神经元在一个周期内的振荡
    
    %% choose data
    classNum = length(inputCell);
    %% PSP 预先计算，加快速度
    allPSPTrain = zeros(classNum,sysinput, nt); %存储所有输入的PSP
    for num=1:classNum                                            %遍历每个单词的输入
        input = inputCell{num};                                  %取出每个样本
        %% input对应的PSP
        for t = 0:dt:nt
            tempInput = t - input;
            tempInput(tempInput<=0) = inf;
            allPSPTrain(num,:, int32(t/dt)+1) = Vnorm* sum( (exp(-tempInput/Tau_m)-exp(-tempInput/Tau_s)), 2);
        end
    end
    incorrectOut = [];
    incorrectLengthList = [0.5,0.1,0.01];
    sampleNum = 1*classNum;
    RightFlag = 0;
    for epoch = 1:maxEpoch                               %遍历固定的epoch
        chooseActCount = zeros(columnNum,classNum);
        %% 存储膜电压和柱的输出
        AllV = zeros(nt*classNum,columnNum);%存储所有柱在一个周期的膜电压
        CA3IV = zeros(nt*classNum,columnNum);
        refV = zeros(nt*classNum,columnNum);
        AllTout = zeros(columnNum,1);
        %% 计算膜电压
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
            %% peak T=100 update weight
            %% CA3层在该周期的输出
            fireTime = toutCA3E;
            if(size(fireTime,2)>1)
                fireTime = sort(fireTime,2);
                for i = size(fireTime,2):-1:1
                    if all(fireTime(:,i) == Inf)
                        fireTime(:,i)=[];
                    else
                        break;
                    end
                end
            end
            %% 近10个单词内被重复利用超过30%，不选择
            fireNeu = find(fireNeurons >= 1);
            blackNeu = [];
            if (word/blackRedius > 1)
                haveFire = sum(chooseActCount(fireNeu,word-blackRedius:word-1),2);
                blackNeuIdx = find(haveFire>=blackTheta);
                blackNeu = fireNeu(blackNeuIdx);
                if ~isempty(blackNeuIdx)
                    fireNeu(blackNeuIdx)=[];
                end
            end
            
            unfiredNeurons = blackNeu;
            %% 选择激活柱
            addNeu = [];
            addNeuTime = [];
            if (length(fireNeu)<activeNum)
                %% 若点火的神经元不到activeNum个
                currentV = AllV(tIdx-period+1:tIdx,:);
                currentV(:,blackNeu) = min(currentV(:));
                [maxV,maxT] = max(currentV); %maxT in [0,T];
                maxV(blackNeu) = min(maxV);
                [maxV,maxIdx] = sort(maxV,'descend');
                fireLen = length(fireNeu);
                addNeu = maxIdx(fireLen+1:activeNum);
                addNeuTime = (maxT(addNeu)+tIdx-period)*dt; %addNeuTime in current period
                chooseActCount(addNeu,word) = 1;
            elseif (length(fireNeu)>activeNum)
                %% 若点火的神经元超过activeNum个
                unfiredLen = length(fireNeu) - activeNum;
                unfireTime =fireTime(fireNeu,1);            %unfireTime in current period
                [unfireTime,unfireIdx] = sort(unfireTime);
                unfiredNeurons(end+1:end+unfiredLen) = fireNeu(unfireIdx(activeNum+1:activeNum+unfiredLen));
                fireNeu = fireNeu(unfireIdx(1:activeNum));
            end
            
            %%                STDP
            %%              调整不够的权重
            if ~isempty(addNeuTime)
                toutTime = addNeuTime;
                for j = 1:size(input,2)
                    tempPlus = toutTime - input(:,j);
                    tempPlus(tempPlus<=0) = Inf; %输入的时间在点火时间之后的设inf
                    deltaWPlus =  learnRatePos*exp(-tempPlus/tau);
                    
                    tempMinus = toutTime - input(:,j);
                    tempMinus(tempMinus>=0) = -Inf;%输入的时间在点火时间之前的设-inf，不降低
                    deltaWMinus = -learnRateNeg*exp(tempMinus/tau);
                    
                    deltaW = deltaWPlus+deltaWMinus;
                    inputWeight(:,addNeu) = inputWeight(:,addNeu)+deltaW;
                end
            end
            
            %%   降低权重
            if ~isempty(unfiredNeurons)
                unfireTime = fireTime(unfiredNeurons,:);
                for i = 1:size(unfireTime,2)
                    toutTime = unfireTime(:,i);
                    toutTime(toutTime==Inf) = 0;
                    for j = 1:size(input,2)
                        tempMinus = toutTime' - input(:,j);
                        tempMinus(tempMinus<=0) = Inf;%输入的时间在点火时间之后的设inf
                        unfDeltaW = -learnRateNeg*exp(-tempMinus/tau);
                        inputWeight(:,unfiredNeurons) = inputWeight(:,unfiredNeurons)+unfDeltaW;
                    end
                end
            end
            
            
            %% 找到一次都没有点火的神经元增加权重
            chooseActCount(fireNeu,word) = 1;
            if word == classNum
                haveFire = sum(chooseActCount,2);
                unchangedNeuron = find(haveFire == 0);
                inputWeight(:,unchangedNeuron) = inputWeight(:,unchangedNeuron)+learnRatePos;
            end
            %% 限制权重的最大值/最小检查权重是否高于最大值或者低于最小值
            %         inputWeight(inputWeight>maxWeight) = maxWeight;
            %         inputWeight(inputWeight<minWeight) = minWeight;
            
        end
        % drawnow;
        
        startTime = through;
        [allFireNeurons,~] = find(AllTout>=startTime);
        allFireNeurons = unique(allFireNeurons);
        allFireOut = AllTout(allFireNeurons,:);
        allFireOut = sort(allFireOut,2,'descend');
        %% 去除全为0的列（没有输出）
        for i = 1:size(allFireOut,2)
            if all(allFireOut(:,i)==0)
                allFireOut(:,i:end) = [];
                break;
            end
        end
        fprintf('Epoch:%d, use columns:%d\n',epoch,length(allFireNeurons));
        %% 计算每个sample的输出
        outputNeuron = {};
        incorrectOutput = zeros(classNum,1);
        for i = 1:classNum
            tout = allFireOut;
            if i~=classNum
                [row,~] = find(tout>=startTime+nt*(i-1)&tout<startTime+nt*i);
            else
                [row,~] = find(tout>=startTime+nt*(i-1));
            end
            neuron = unique(row);
            outputNeuron{i} = neuron;
            if length(neuron)~=activeNum
                %         fprintf('sample %d-neuron %d, at epoch %d\n',i,length(neuron),epoch);
                incorrectOutput(i) = length(neuron);
            end
        end
        incorrectAccuracy = length(find(incorrectOutput~=0))/sampleNum;
        fprintf('%f sample is not fire with active numbers \n',incorrectAccuracy);
        incorrectOut(epoch) = incorrectAccuracy;
        if ~isempty(incorrectLengthList)&&incorrectAccuracy<=incorrectLengthList(1)
            incorrectLengthList(1) = [];
            learnRatePos = learnRatePos*0.5;
            learnRateNeg =  learnRateNeg*0.5;
        end
        if incorrectAccuracy==0
            RightFlag = RightFlag+1;
            if  RightFlag >=2
                break;
            end
        end
    end
    AllEpoch(trial) = epoch;
    outputColumn = zeros(classNum,10);
    for i = 1:classNum
        outputColumn(i,:) = outputNeuron{i};
    end
    outputTimeCal
end