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
%% ����Ĥ��ѹ���������
AllV = zeros(nt*classNum,columnNum);         %�洢��������һ�����ڵ�Ĥ��ѹ
CA3IV = zeros(nt*classNum,columnNum);
refV = zeros(nt*classNum,columnNum);
AllTout = zeros(columnNum,1);
multiOut = {};
multiColumn = [];
multiNum = 1;
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
