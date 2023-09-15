clear;
close;
%% �ռ��-�����
%% ��������
load('data1000.mat');                            %������Ӧ������
trialNum = 1;

AllEpoch = zeros(1:trialNum);
for trial = 1:trialNum
    learnRatePos = 0.0015;
    learnRateNeg = 0.001;
    columnNum = 512;
    activeNum = 10;
    maxEpoch = 200;                               %ѵ����epoch����
    sysinput = 1000;
    inputWeight = normrnd(0.01,0.01,sysinput,columnNum); %Ȩ�س�ʼ����ʽ����˹���
    weight_start = inputWeight;                          %�����ʼȨ��
    %% STDP para
    maxWeight = 1/activeNum;          %% ���Ȩ��
    minWeight = -1/activeNum;         %% ��СȨ��
    tau=25;               %% STDP�Ĳ���
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
    allOsc = inputCA3;  %����ÿ����Ԫ��һ�������ڵ���
    
    %% choose data
    classNum = length(inputCell);
    %% PSP Ԥ�ȼ��㣬�ӿ��ٶ�
    allPSPTrain = zeros(classNum,sysinput, nt); %�洢���������PSP
    for num=1:classNum                                            %����ÿ�����ʵ�����
        input = inputCell{num};                                  %ȡ��ÿ������
        %% input��Ӧ��PSP
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
    for epoch = 1:maxEpoch                               %�����̶���epoch
        chooseActCount = zeros(columnNum,classNum);
        %% �洢Ĥ��ѹ���������
        AllV = zeros(nt*classNum,columnNum);%�洢��������һ�����ڵ�Ĥ��ѹ
        CA3IV = zeros(nt*classNum,columnNum);
        refV = zeros(nt*classNum,columnNum);
        AllTout = zeros(columnNum,1);
        %% ����Ĥ��ѹ
        for word = 1:sampleNum
            startTime = (word-1)*nt;
            input = startTime+inputCell{word};                     %ȡ��ÿ������
            intTime = Inf*ones(columnNum,1);                       %������Ԫ�����
            toutCA3E = Inf*ones(columnNum,1);                     %���ڸ����ڵ����
            toutCa3Last = -Inf*ones(columnNum,1);                  %���ڸ��������һ�ε����
            fireNeurons = zeros(columnNum,1);                      %��Щ�����
            for t = startTime+(1:nt)*dt                         %�����Ԫ�ĵ��ʱ�䣬����0��Tmax
                tIdx = t/dt;
                %% ����ǰ������Ĳ�����PSP�Լ�ǰ����ѹ
                currentT = tIdx-startTime;
                PSPInput = allPSPTrain(word,:,currentT)';
                inputV = inputWeight'*PSPInput;
                %% ��Ӧ��
                ref = -exp((toutCa3Last-t)/Tau_a);
                %% ����PSP
                tempInh = t - intTime;
                tempInh(tempInh<=0) = inf;
                CA3I = -ca3value*sum(exp(-tempInh/Tau_s),2);
                %% ��������Ĥ��ѹ
                CA3EV = inputV + CA3I + ref+allOsc(:,currentT);
                %% ��¼
                AllV(tIdx,:) = CA3EV;
                CA3IV(tIdx,:) = CA3I;
                refV(tIdx,:) = ref;
                %% ���ڳ�����ֵ����Ԫ
                neuron = find(CA3EV >=Theta);      %Ĥ��ѹ�Ƿ������ֵ���
                if(length(neuron))                 %��������Ԫ���
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
            %% CA3���ڸ����ڵ����
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
            %% ��10�������ڱ��ظ����ó���30%����ѡ��
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
            %% ѡ�񼤻���
            addNeu = [];
            addNeuTime = [];
            if (length(fireNeu)<activeNum)
                %% ��������Ԫ����activeNum��
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
                %% ��������Ԫ����activeNum��
                unfiredLen = length(fireNeu) - activeNum;
                unfireTime =fireTime(fireNeu,1);            %unfireTime in current period
                [unfireTime,unfireIdx] = sort(unfireTime);
                unfiredNeurons(end+1:end+unfiredLen) = fireNeu(unfireIdx(activeNum+1:activeNum+unfiredLen));
                fireNeu = fireNeu(unfireIdx(1:activeNum));
            end
            
            %%                STDP
            %%              ����������Ȩ��
            if ~isempty(addNeuTime)
                toutTime = addNeuTime;
                for j = 1:size(input,2)
                    tempPlus = toutTime - input(:,j);
                    tempPlus(tempPlus<=0) = Inf; %�����ʱ���ڵ��ʱ��֮�����inf
                    deltaWPlus =  learnRatePos*exp(-tempPlus/tau);
                    
                    tempMinus = toutTime - input(:,j);
                    tempMinus(tempMinus>=0) = -Inf;%�����ʱ���ڵ��ʱ��֮ǰ����-inf��������
                    deltaWMinus = -learnRateNeg*exp(tempMinus/tau);
                    
                    deltaW = deltaWPlus+deltaWMinus;
                    inputWeight(:,addNeu) = inputWeight(:,addNeu)+deltaW;
                end
            end
            
            %%   ����Ȩ��
            if ~isempty(unfiredNeurons)
                unfireTime = fireTime(unfiredNeurons,:);
                for i = 1:size(unfireTime,2)
                    toutTime = unfireTime(:,i);
                    toutTime(toutTime==Inf) = 0;
                    for j = 1:size(input,2)
                        tempMinus = toutTime' - input(:,j);
                        tempMinus(tempMinus<=0) = Inf;%�����ʱ���ڵ��ʱ��֮�����inf
                        unfDeltaW = -learnRateNeg*exp(-tempMinus/tau);
                        inputWeight(:,unfiredNeurons) = inputWeight(:,unfiredNeurons)+unfDeltaW;
                    end
                end
            end
            
            
            %% �ҵ�һ�ζ�û�е�����Ԫ����Ȩ��
            chooseActCount(fireNeu,word) = 1;
            if word == classNum
                haveFire = sum(chooseActCount,2);
                unchangedNeuron = find(haveFire == 0);
                inputWeight(:,unchangedNeuron) = inputWeight(:,unchangedNeuron)+learnRatePos;
            end
            %% ����Ȩ�ص����ֵ/��С���Ȩ���Ƿ�������ֵ���ߵ�����Сֵ
            %         inputWeight(inputWeight>maxWeight) = maxWeight;
            %         inputWeight(inputWeight<minWeight) = minWeight;
            
        end
        % drawnow;
        
        startTime = through;
        [allFireNeurons,~] = find(AllTout>=startTime);
        allFireNeurons = unique(allFireNeurons);
        allFireOut = AllTout(allFireNeurons,:);
        allFireOut = sort(allFireOut,2,'descend');
        %% ȥ��ȫΪ0���У�û�������
        for i = 1:size(allFireOut,2)
            if all(allFireOut(:,i)==0)
                allFireOut(:,i:end) = [];
                break;
            end
        end
        fprintf('Epoch:%d, use columns:%d\n',epoch,length(allFireNeurons));
        %% ����ÿ��sample�����
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