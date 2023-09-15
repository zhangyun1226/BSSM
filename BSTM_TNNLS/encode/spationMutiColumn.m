clear;
close;
%% �ռ��-�����
%% ��������
load('data.mat');                            %������Ӧ������
initParameter;                               %��ʼ���Ĳ���
columnNumList = [512,1024,1536,2048,2560];   %���������б�
maxSimilityList = [0.3,0.3,0.3,0.3,0.3];     %�������ʼ����ı�ʾ��������ƶ�
maxEpoch = 100;                               %ѵ����epoch����
maxCount = 100;                               %ÿ������౻ѡ�������,500 25
wordNum = max(sequence(:));                  %���ʸ���/��������
%% STDP para
tau = 20;               %% STDP�Ĳ���
maxWeight = 1;          %% ���Ȩ��
minWeight = -1;         %% ��СȨ��
learnRatePos = 0.005;    %% STDP��ѧϰ��
learnRateNeg = 0.0005;   %% STDP��ѧϰ��
%% PSP Ԥ�ȼ��㣬�ӿ��ٶ�
allPSPTrain = zeros(wordNum,sysinput, length(periodicTime)); %�洢���������PSP
for num=1:wordNum                                            %����ÿ�����ʵ�����
    input = inputCell{num};                                  %ȡ��ÿ������
    %% input��Ӧ��PSP
    for t = periodicTime
        tempInput = t - input;
        tempInput(tempInput<=0) = inf;
        allPSPTrain(num,:, int32(t/T_step)+1) = Vnorm* sum( (exp(-tempInput/Tau_m)-exp(-tempInput/Tau_s)), 2);
    end
end
%% ѧϰ
for num = 1:length(columnNumList)            %���������б�
    maxSimility = maxSimilityList(num);                  %��ͬ���ʼ�ѡ�񵥴ʵ�������ƶ�
    columnNum = columnNumList(num);                      %���ĸ���
    inputWeight = normrnd(0.05,0.05,sysinput,columnNum); %Ȩ�س�ʼ����ʽ����˹���
    weight_start = inputWeight;                          %�����ʼȨ��
    activeColumnAll = zeros(wordNum,activeNum);          %��¼��ǰepoch,ÿ������/����ѡ��ļ�����
    lastActiveColumn = zeros(wordNum,activeNum);         %��¼��һ��epochÿ������ѡ��ļ�����
    activeToutAll  = Inf*ones(wordNum,activeNum);        %ѵ��ÿ�����ʵ����
    fprintf('columnNum %d\n',columnNum);
    %% ѵ��ÿ�����ʣ�ʹ��ͬ�ĵ����ò�ͬ������𣬲���¼����ʱ�����Ӧ��������
    for epoch = 1:maxEpoch                               %�����̶���epoch
        countActiveColumn = zeros(columnNum,1);          %ÿ������ѡ��Ĵ���
        lastActiveColumn = activeColumnAll;              %��¼��һ��epochÿ������ѡ��ļ�����
        activeColumnAll = zeros(wordNum,activeNum);      %��ǰepoch�ļ�������ʾ����
        hasNoOutNum = 0;                                 %ѡ��ļ�����û������ĸ��������ΪINF��
        for word = 1:wordNum                             %����ÿ�����ʵ����룬ȥѡ�񼤻���
            if mod(word,1000) == 0                       %ÿ1000�����һ��
                word
            end
            %% �ռ����ϸ
            input = inputCell{word};                     %ȡ��ÿ������
            %% ����Ĥ��ѹ���������
            toutColumn = Inf*ones(columnNum,1);          %�������
            ref = zeros(columnNum,1);                    %���Ĳ�Ӧ�ڣ�ֻ�õ��һ�Σ�����Ӧ�ڱ�Ϊ-inf���������ٵ��
            AllV = zeros(length(periodicTime),columnNum);%�洢��������һ�����ڵ�Ĥ��ѹ
            for t = periodicTime                         %�����Ԫ�ĵ��ʱ�䣬����0��Tmax
                %% ����ǰ������Ĳ�����PSP
                PSP = allPSPTrain(word,:, int32(t/T_step)+1)';
                inputV = inputWeight'*PSP;
                %% ��������Ĥ��ѹ
                V = inputV + ref;
                AllV(int32(t/T_step)+1,:) = V;       %��¼��Ĥ��ѹ
                neuron = find(V >=Theta);            %Ĥ��ѹ�Ƿ������ֵ���
                if(length(neuron))                   %��������Ԫ���
                    toutColumn(neuron) = t;          %�����������Ϊ��ǰʱ��
                    ref(neuron) = -Inf;              %��Ӧ������Ϊ����
                end
            end
            %% �洢��ǰÿ���������������0-1��ά����
            toutColumnNum = zeros(columnNum,1);
            toutColumnNum(toutColumn ~= Inf) = 1;
            %% �������Ĳ���1,ÿ�������ܱ�̫��ѡ������̫��ѡ��������������
            blackColumns = find(countActiveColumn>=maxCount); %��������-���ܱ�ѡ�����
            toutColumnNum(blackColumns) = 0; %������������������Ϊ0
            toutColumn(blackColumns,:) = Inf;%������������������
            AllV(:,blackColumns) = min(AllV(:));%��������Ĥ��ѹ���
            %% �������Ĳ���2��
            ...1������ѡ��������ļ�������
            ...2��������ļ�����̫��ʱ��ѡ�����缤���������Ĥ��ѹ������
            ...3��û�����ʱ��ѡ��Ĥ��ѹ�����Ǽ���
            toutMaxNum = 1;                  %����������Ϊ1
            activeState = [];                %��ǰ���ʵļ�����
            candidateNum = activeNum;        %��ѡ�������ĸ������ʼΪ������������
            if toutMaxNum~=0                 %��������������
                columns = find(toutColumnNum == toutMaxNum);  %�ҵ������������
                activeLen = length(columns);                  %���������������Ŀ
                if activeLen >= candidateNum                  %��������Ŀ������ѡ��
                    toutEarly = toutColumn(columns);          %����������ļ������ʱ��
                    [toutEarly,earlyIndex] = sort(toutEarly); %�����ʱ������������Щ��
                    uniqueTout = unique(toutEarly);           %ȥ�غ�����
                    for i = 1:length(uniqueTout)
                        currentFire = uniqueTout(i);          %�����˳���ȡ���ʱ��
                        currentFireIndex = find(toutEarly == currentFire); %�ڵ�ǰʱ�����������Щ
                        if length(currentFireIndex)<candidateNum           %����ǰ��������ȫ�������ѡ��
                            candidateColumn = columns(earlyIndex(currentFireIndex));%�����Щ�������λ��
                            activeState(end+1:end+length(candidateColumn)) = candidateColumn;%���뼤����
                            candidateNum = candidateNum - length(candidateColumn);  %���º�ѡ������Ŀ
                        else %����ǰʱ��������պû����ж��࣬�Ƚ�Ĥ��ѹ��Ĥ��ѹ���ĺ�ѡ�����ļ��������뼤�����б�
                            currentAllV = AllV(:,earlyIndex(currentFireIndex));%��ȡ��Щ����Ĥ��ѹ
                            currentAllV=max(currentAllV);                      %������Щ�������Ĥ��ѹ
                            [~,currentcolumns] = sort(currentAllV,'descend');  %�����Ĥ��ѹ���ս�������
                            currentcolumns = currentFireIndex(currentcolumns); %��ȡ��Ӧ������
                            currentcolumns = currentcolumns(1:candidateNum);   %ֻ������ѡ�������ļ�����
                            activeState(end+1:end+candidateNum) = columns(earlyIndex(currentcolumns));%���뼤����
                            break;
                        end
                    end
                    candidateNum = 0; %������ѡ�����
                else %��������Ŀ���ں�ѡ��������
                    activeState(end+1:end+activeLen) = columns; %ȫ�������ѡ��
                    candidateNum = candidateNum - activeLen;    %���º�ѡ������Ŀ
                end
            end
            %�жϼ������Ƿ���ѡ��ϣ�==0��û����ɣ�֤��ʣ�µĶ���δ���ģ�ѡ��Ĥ��ѹ�����Ǽ���
            if (candidateNum ~= 0)          %��������ʣ����δѡ��
                AllV(:,activeState) = -Inf; %�޳���������Ĥ��ѹ�����ǿ϶�������
                maxV=max(AllV);             %����ÿ���������Ĥ��ѹ
                [maxV,columns] = sort(maxV,'descend');
                if maxV(1) == maxV(end)
                    fprintf('cant choose the column!!!,the column is not enough\n');
                else
                    activeState(end+1:end+candidateNum) = columns(1:candidateNum);
                end
            end
            %% �������Ĳ���3�����������ƶȹ��ߵ�������ȥ���غϵĲ��ּ�����������ѡ��ֱ���������ƶȺϸ�
            chooseAgain = 1;            %�Ƿ���Ҫ����ѡ����
            minV = min(AllV(:));        %��СĤ��ѹ
            while chooseAgain == 1      %��Ҫ����ѡ��
                %% 3.1 ������ǰ����ѡ������͵�ǰ����ѡ������ƶȣ�������������ƶȣ���Ҫ����ѡ��
                chooseAgain = 0;        %Ĭ����������ѡ��
                for i = 1:word-1        %��ȡ��ǰ������ǰ�����м�����
                    activei = activeColumnAll(i,:);               %��i�����ʵļ�����
                    %% ��ǰ���ʵļ������͵�i�����ʵļ������Ľ���
                    diffNeurons = activeState-activei';           %��i�����ʵļ������͵�ǰ������֮��
                    [~,comIdx] = find(diffNeurons==0);            %Ϊ0��ʾ����ͬ��Ԫ��
                    interActive = activeState(comIdx);            %�ҵ���ͬ��Ԫ��
                    simility = length(interActive)/activeNum;     %�������ƶ�
                    if simility >maxSimility                      %�����ƶȳ���������ƶ�
                        chooseAgain = 1;                          %��Ҫ����ѡ��
                        tempblack = interActive(1:round(length(interActive)/2));%�ҳ���������һ��
                        blackColumns(end+1:end+length(tempblack)) = tempblack;  %��Щ����������������������ǿ��ģ���ʵ�������δ��
                        toutColumnNum(tempblack) = 0;   %������������������Ϊ0
                        toutColumn(tempblack,:) = Inf;  %������������������
                        AllV(:,tempblack) = minV;       %��������Ĥ��ѹ���
                    end
                end
                %% 3.2 ����ѡ�񼤻���-�������2
                if chooseAgain == 1
                    toutMaxNum = 1;                  %����������Ϊ1
                    activeState = [];                %��ǰ���ʵļ�����
                    candidateNum = activeNum;        %��ѡ�������ĸ������ʼΪ������������
                    if toutMaxNum~=0                 %��������������
                        columns = find(toutColumnNum == toutMaxNum);  %�ҵ������������
                        activeLen = length(columns);                  %���������������Ŀ
                        if activeLen >= candidateNum                  %��������Ŀ������ѡ��
                            toutEarly = toutColumn(columns);          %����������ļ������ʱ��
                            [toutEarly,earlyIndex] = sort(toutEarly); %�����ʱ������������Щ��
                            uniqueTout = unique(toutEarly);           %ȥ�غ�����
                            for i = 1:length(uniqueTout)
                                currentFire = uniqueTout(i);          %��ʱ��˳���ȡ���ʱ��
                                currentFireIndex = find(toutEarly == currentFire); %�ڵ�ǰ����������Щ
                                if length(currentFireIndex)<candidateNum           %����ǰ��������ȫ�������ѡ��
                                    candidateColumn = columns(earlyIndex(currentFireIndex));%�����Щ�������λ��
                                    activeState(end+1:end+length(candidateColumn)) = candidateColumn;%���뼤����
                                    candidateNum = candidateNum - length(candidateColumn);  %���º�ѡ������Ŀ
                                else %����ǰʱ��������պû����ж��࣬�Ƚ�Ĥ��ѹ��Ĥ��ѹ���ĺ�ѡ�����ļ��������뼤�����б�
                                    currentAllV = AllV(:,earlyIndex(currentFireIndex));%��ȡ��Щ����Ĥ��ѹ
                                    currentAllV=max(currentAllV);                      %������Щ�������Ĥ��ѹ
                                    [~,currentcolumns] = sort(currentAllV,'descend');  %�����Ĥ��ѹ���ս�������
                                    currentcolumns = currentFireIndex(currentcolumns); %��ȡ��Ӧ������
                                    currentcolumns = currentcolumns(1:candidateNum);   %ֻ������ѡ�������ļ�����
                                    activeState(end+1:end+candidateNum) = columns(earlyIndex(currentcolumns));%���뼤����
                                    break;
                                end
                            end
                            candidateNum = 0; %������ѡ�����
                        else %��������Ŀ���ں�ѡ��������
                            activeState(end+1:end+activeLen) = columns; %ȫ�������ѡ��
                            candidateNum = candidateNum - activeLen;    %���º�ѡ������Ŀ
                        end
                    end
                    %�жϼ������Ƿ���ѡ��ϣ�==0��û����ɣ�֤��ʣ�µĶ���δ���ģ�ѡ��Ĥ��ѹ�����Ǽ���
                    if (candidateNum ~= 0)          %��������ʣ����δѡ��
                        AllV(:,activeState) = -Inf; %�޳���������Ĥ��ѹ�����ǿ϶�������
                        maxV=max(AllV);             %����ÿ���������Ĥ��ѹ
                        [maxV,columns] = sort(maxV,'descend');
                        if maxV(1) == maxV(end)
                            fprintf('cant choose the column!!!,the column is not enough\n');
                        else
                            activeState(end+1:end+candidateNum) = columns(1:candidateNum);
                        end
                    end
                end
            end
            %% ��ǰ������ѡ����ϣ���ѡ��ļ������ļ�������1�����洢������������
            countActiveColumn(activeState) = countActiveColumn(activeState) +1;
            activeState = sort(activeState);        %��������index����С��������
            activeColumnAll(word,:) = activeState;  %�洢��ǰ������
            activeTout = toutColumn(activeState,:); %��ȡ�����������
            activeToutAll(word,:) = activeTout;     %�洢��ǰ�����������
            if (any(activeTout == Inf))             %�Ƿ����û���������
                hasNoOutNum = hasNoOutNum + 1 ;     %������1 
            end
            %% STDP learn �������뵽�����������ӣ�ǰ�����ӣ�
            for activeIndex = 1:activeNum          %����ÿ��������
                activeColumn = activeState(activeIndex);
                fireTime = toutColumn(activeColumn);%��ȡÿ�����ļ���ʱ��
                fireTime(fireTime == Inf) = [];     %ȥ�������ʱ���
                if(isempty(fireTime))
                    fireTime = Tmax;                            %û��������������Ϊ���ĵ��ʱ��
                end
                %% STDP �кܶ��ظ��ļ���ʱ�䣬�������Ż�
                for j = 1:length(fireTime)
                    %% �����ʱ���ڵ��ʱ��֮�󣬽���Ȩ��
                    tempMinus = fireTime(j) - input;
                    tempMinus(tempMinus>=0) = -Inf;%�����ʱ���ڵ��ʱ��֮ǰ����-inf��������
                    deltaWMinus = -learnRateNeg*sum(exp(tempMinus/tau),2);
                    %% �����ʱ���ڵ��ʱ��֮ǰ������Ȩ��
                    tempPlus = fireTime(j) - input;
                    tempPlus(tempPlus == -Inf) = Inf;%�����ʱ���ڵ��ʱ��֮�����inf��������Ȩ��
                    tempPlus(tempPlus<=0) = Inf;
                    deltaPlus =  learnRatePos*sum(exp(-tempPlus/tau),2);
                    if activeTout(activeIndex) == Inf
                        deltaPlus = deltaPlus*1.2;
                    end
                    %% Ȩ�ظñ���Ϊ���Ӻͽ��͵��ܺ�
                    deltaW = deltaWMinus + deltaPlus;
                    inputWeight(:,activeColumn) = inputWeight(:,activeColumn)+deltaW;
                end
            end
            %% ������������˵�Ȩ��
            fireColumns = find(toutColumnNum == 1);
            inhColumns = setdiff(fireColumns,activeIndex);
            for j = 1:length(inhColumns)
                inhColumn = inhColumns(j);
                %% �����ʱ���ڵ��ʱ��֮�󣬽���Ȩ��
                fireTime = T_step;
                tempMinus = fireTime - input;
                tempMinus(tempMinus>=0) = -Inf;%�����ʱ���ڵ��ʱ��֮ǰ����-inf��������
                deltaWMinus = -learnRateNeg*sum(exp(tempMinus/tau),2);
                %% Ȩ�ظñ���Ϊ���Ӻͽ��͵��ܺ�
                inputWeight(:,inhColumn) = inputWeight(:,inhColumn)+deltaWMinus;
            end
            
            %% ����Ȩ�ص����ֵ/��С���Ȩ���Ƿ�������ֵ���ߵ�����Сֵ
            inputWeight(inputWeight>maxWeight) = maxWeight;
            inputWeight(inputWeight<minWeight) = minWeight;  
        end
        %% ������һ�ֺ���һ�ָõ���ѡ�����֮������ƶ�
        chooseSimility = zeros(wordNum,1);
        for i = 1:wordNum                  %�������е���
            activei = activeColumnAll(i,:);%��ȡ�õ��ʵļ�����
            if epoch~=1
                oldActivei = lastActiveColumn(i,:);%��ȡ�õ�����һʱ�̵ļ�����
                %�������м������ƶ�
                diffNeurons = activei-oldActivei';
                [~,comIdx] = find(diffNeurons==0);
                chooseSimility(i) = length(comIdx)/activeNum;
            end
        end
        meanSimility = sum(chooseSimility(:))/wordNum;
        fprintf('epoch%d,����һ��ѡ�����С���ƶ�:%f, ƽ�����ƶ�: %f\n',epoch,min(chooseSimility(:)),meanSimility);
        fprintf('active column without output:%d\n',hasNoOutNum);
        %% �����ı�������
        if meanSimility>0.99 %ƽ�����ƶ�Ϊ0.99
            break;
        end
    end
    %% �ٴμ����������֮������ƶ��Ƿ���ȷ
    chooseSimility = zeros(wordNum,wordNum);
    for i = 1:wordNum-1
        activei = activeColumnAll(i,:);
        for j = i+1:wordNum
            oldActivei = activeColumnAll(j,:);
            diffNeurons = activei-oldActivei';
            [~,comIdx] = find(diffNeurons==0);
            chooseSimility(i,j) = length(comIdx)/10;
        end
    end
    fprintf('end!!! max simility of two words %f\n',max(chooseSimility(:)));
    %% �����ÿ�����ʵ����ı�ʾ���Լ�ÿ���������
    columnCell = activeColumnAll;
    columnOut = activeToutAll;
    for i = 1:length(columnOut)
        activeTout = columnOut(i,:);
        idx = find(activeTout == Inf);
        if ~isempty(idx)
            tempOut = activeTout(activeTout~=Inf);
            activeTout(idx) = roundn(mean(tempOut),-1);
        end
        columnOut(i,:) = activeTout;
    end
    currentStr = 'spation';
    currentStr = [currentStr,num2str(columnNum)];
    currentStr = [currentStr,'.mat'];
    save(currentStr, 'columnCell', 'columnOut','chooseSimility','meanSimility');
end


