clear;
close;
%% 空间层-编码层
%% 参数设置
load('data.mat');                            %加载相应的数据
initParameter;                               %初始化的参数
columnNumList = [512,1024,1536,2048,2560];   %柱的数量列表
maxSimilityList = [0.3,0.3,0.3,0.3,0.3];     %两个单词间柱的表示的最大相似度
maxEpoch = 100;                               %训练的epoch数量
maxCount = 100;                               %每个柱最多被选择的数量,500 25
wordNum = max(sequence(:));                  %单词个数/样本个数
%% STDP para
tau = 20;               %% STDP的参数
maxWeight = 1;          %% 最大权重
minWeight = -1;         %% 最小权重
learnRatePos = 0.005;    %% STDP的学习率
learnRateNeg = 0.0005;   %% STDP的学习率
%% PSP 预先计算，加快速度
allPSPTrain = zeros(wordNum,sysinput, length(periodicTime)); %存储所有输入的PSP
for num=1:wordNum                                            %遍历每个单词的输入
    input = inputCell{num};                                  %取出每个样本
    %% input对应的PSP
    for t = periodicTime
        tempInput = t - input;
        tempInput(tempInput<=0) = inf;
        allPSPTrain(num,:, int32(t/T_step)+1) = Vnorm* sum( (exp(-tempInput/Tau_m)-exp(-tempInput/Tau_s)), 2);
    end
end
%% 学习
for num = 1:length(columnNumList)            %遍历柱的列表
    maxSimility = maxSimilityList(num);                  %不同单词间选择单词的最大相似度
    columnNum = columnNumList(num);                      %柱的个数
    inputWeight = normrnd(0.05,0.05,sysinput,columnNum); %权重初始化方式：高斯随机
    weight_start = inputWeight;                          %保存初始权重
    activeColumnAll = zeros(wordNum,activeNum);          %记录当前epoch,每个样本/单词选择的激活柱
    lastActiveColumn = zeros(wordNum,activeNum);         %记录上一个epoch每个样本选择的激活柱
    activeToutAll  = Inf*ones(wordNum,activeNum);        %训练每个单词的输出
    fprintf('columnNum %d\n',columnNum);
    %% 训练每个单词，使不同的单词让不同的柱点火，并记录其点火时间和相应的柱序列
    for epoch = 1:maxEpoch                               %遍历固定的epoch
        countActiveColumn = zeros(columnNum,1);          %每个柱被选择的次数
        lastActiveColumn = activeColumnAll;              %记录上一个epoch每个样本选择的激活柱
        activeColumnAll = zeros(wordNum,activeNum);      %当前epoch的激活柱表示重置
        hasNoOutNum = 0;                                 %选择的激活柱没有输出的个数（输出为INF）
        for word = 1:wordNum                             %遍历每个单词的输入，去选择激活柱
            if mod(word,1000) == 0                       %每1000个输出一次
                word
            end
            %% 空间池详细
            input = inputCell{word};                     %取出每个样本
            %% 计算膜电压和柱的输出
            toutColumn = Inf*ones(columnNum,1);          %柱的输出
            ref = zeros(columnNum,1);                    %柱的不应期，只让点火一次，点火后不应期变为-inf，不让其再点火
            AllV = zeros(length(periodicTime),columnNum);%存储所有柱在一个周期的膜电压
            for t = periodicTime                         %算该神经元的点火时间，遍历0到Tmax
                %% 计算前馈输入的产生的PSP
                PSP = allPSPTrain(word,:, int32(t/T_step)+1)';
                inputV = inputWeight'*PSP;
                %% 计算柱的膜电压
                V = inputV + ref;
                AllV(int32(t/T_step)+1,:) = V;       %记录下膜电压
                neuron = find(V >=Theta);            %膜电压是否大于阈值点火
                if(length(neuron))                   %若存在神经元点火
                    toutColumn(neuron) = t;          %它的输出设置为当前时间
                    ref(neuron) = -Inf;              %不应期设置为无穷
                end
            end
            %% 存储当前每个柱的输出个数，0-1二维矩阵
            toutColumnNum = zeros(columnNum,1);
            toutColumnNum(toutColumn ~= Inf) = 1;
            %% 激活柱的策略1,每个柱不能被太多选择，若被太多选择则加入黑明单了
            blackColumns = find(countActiveColumn>=maxCount); %黑明单柱-不能被选择的柱
            toutColumnNum(blackColumns) = 0; %黑名单的输出个数清除为0
            toutColumn(blackColumns,:) = Inf;%黑名单的输出个数清空
            AllV(:,blackColumns) = min(AllV(:));%黑名单的膜电压清除
            %% 激活柱的策略2：
            ...1）优先选择有输出的激活柱；
            ...2）有输出的激活柱太多时，选择最早激活的柱或者膜电压最大的柱
            ...3）没有输出时，选择膜电压最大的那几个
            toutMaxNum = 1;                  %最大输出个数为1
            activeState = [];                %当前单词的激活柱
            candidateNum = activeNum;        %候选激活柱的个数，最开始为激活柱的数量
            if toutMaxNum~=0                 %如果有神经柱有输出
                columns = find(toutColumnNum == toutMaxNum);  %找到有输出的神经柱
                activeLen = length(columns);                  %有输出的神经柱的数目
                if activeLen >= candidateNum                  %若激活数目超过候选柱
                    toutEarly = toutColumn(columns);          %找它们最早的激活输出时间
                    [toutEarly,earlyIndex] = sort(toutEarly); %按点火时间升序排列这些柱
                    uniqueTout = unique(toutEarly);           %去重后的输出
                    for i = 1:length(uniqueTout)
                        currentFire = uniqueTout(i);          %按点火顺序获取点火时间
                        currentFireIndex = find(toutEarly == currentFire); %在当前时间点火的柱有哪些
                        if length(currentFireIndex)<candidateNum           %若当前点火的柱可全部放入候选柱
                            candidateColumn = columns(earlyIndex(currentFireIndex));%获得这些具体的柱位置
                            activeState(end+1:end+length(candidateColumn)) = candidateColumn;%加入激活柱
                            candidateNum = candidateNum - length(candidateColumn);  %更新候选柱的数目
                        else %若当前时间点火的柱刚好或者有多余，比较膜电压，膜电压最大的候选个数的激活柱加入激活柱列表
                            currentAllV = AllV(:,earlyIndex(currentFireIndex));%获取这些柱的膜电压
                            currentAllV=max(currentAllV);                      %计算这些柱的最大膜电压
                            [~,currentcolumns] = sort(currentAllV,'descend');  %对最大膜电压按照降序排列
                            currentcolumns = currentFireIndex(currentcolumns); %获取对应的柱号
                            currentcolumns = currentcolumns(1:candidateNum);   %只保留候选柱数量的激活柱
                            activeState(end+1:end+candidateNum) = columns(earlyIndex(currentcolumns));%加入激活柱
                            break;
                        end
                    end
                    candidateNum = 0; %激活柱选择完毕
                else %若激活数目低于候选柱的数量
                    activeState(end+1:end+activeLen) = columns; %全部加入候选柱
                    candidateNum = candidateNum - activeLen;    %更新候选柱的数目
                end
            end
            %判断激活柱是否挑选完毕，==0，没有完成，证明剩下的都是未点火的，选择膜电压最大的那几个
            if (candidateNum ~= 0)          %激活柱还剩有柱未选择
                AllV(:,activeState) = -Inf; %剔除激活柱的膜电压，它们肯定是最大的
                maxV=max(AllV);             %计算每个柱的最大膜电压
                [maxV,columns] = sort(maxV,'descend');
                if maxV(1) == maxV(end)
                    fprintf('cant choose the column!!!,the column is not enough\n');
                else
                    activeState(end+1:end+candidateNum) = columns(1:candidateNum);
                end
            end
            %% 激活柱的策略3：若存在相似度过高的神经柱，去掉重合的部分激活柱，重新选择，直到柱的相似度合格
            chooseAgain = 1;            %是否需要重新选择柱
            minV = min(AllV(:));        %最小膜电压
            while chooseAgain == 1      %需要重新选择
                %% 3.1 计算以前单词选择的柱和当前单词选择的相似度，若超过最大相似度，需要重新选择
                chooseAgain = 0;        %默认无需重新选择
                for i = 1:word-1        %获取当前单词以前的所有激活柱
                    activei = activeColumnAll(i,:);               %第i个单词的激活柱
                    %% 当前单词的激活柱和第i个单词的激活柱的交集
                    diffNeurons = activeState-activei';           %第i个单词的激活柱和当前激活柱之差
                    [~,comIdx] = find(diffNeurons==0);            %为0表示有相同的元素
                    interActive = activeState(comIdx);            %找到相同的元素
                    simility = length(interActive)/activeNum;     %计算相似度
                    if simility >maxSimility                      %若相似度超过最大相似度
                        chooseAgain = 1;                          %需要重新选择
                        tempblack = interActive(1:round(length(interActive)/2));%找出交集柱的一半
                        blackColumns(end+1:end+length(tempblack)) = tempblack;  %这些交集柱加入黑名单，给我们看的，真实这个参数未用
                        toutColumnNum(tempblack) = 0;   %黑名单的输出个数清除为0
                        toutColumn(tempblack,:) = Inf;  %黑名单的输出个数清空
                        AllV(:,tempblack) = minV;       %黑名单的膜电压清除
                    end
                end
                %% 3.2 重新选择激活柱-激活策略2
                if chooseAgain == 1
                    toutMaxNum = 1;                  %最大输出个数为1
                    activeState = [];                %当前单词的激活柱
                    candidateNum = activeNum;        %候选激活柱的个数，最开始为激活柱的数量
                    if toutMaxNum~=0                 %如果有神经柱有输出
                        columns = find(toutColumnNum == toutMaxNum);  %找到有输出的神经柱
                        activeLen = length(columns);                  %有输出的神经柱的数目
                        if activeLen >= candidateNum                  %若激活数目超过候选柱
                            toutEarly = toutColumn(columns);          %找它们最早的激活输出时间
                            [toutEarly,earlyIndex] = sort(toutEarly); %按点火时间升序排列这些柱
                            uniqueTout = unique(toutEarly);           %去重后的输出
                            for i = 1:length(uniqueTout)
                                currentFire = uniqueTout(i);          %按时间顺序获取点火时间
                                currentFireIndex = find(toutEarly == currentFire); %在当前点火的柱有哪些
                                if length(currentFireIndex)<candidateNum           %若当前点火的柱可全部放入候选柱
                                    candidateColumn = columns(earlyIndex(currentFireIndex));%获得这些具体的柱位置
                                    activeState(end+1:end+length(candidateColumn)) = candidateColumn;%加入激活柱
                                    candidateNum = candidateNum - length(candidateColumn);  %更新候选柱的数目
                                else %若当前时间点火的柱刚好或者有多余，比较膜电压，膜电压最大的候选个数的激活柱加入激活柱列表
                                    currentAllV = AllV(:,earlyIndex(currentFireIndex));%获取这些柱的膜电压
                                    currentAllV=max(currentAllV);                      %计算这些柱的最大膜电压
                                    [~,currentcolumns] = sort(currentAllV,'descend');  %对最大膜电压按照降序排列
                                    currentcolumns = currentFireIndex(currentcolumns); %获取对应的柱号
                                    currentcolumns = currentcolumns(1:candidateNum);   %只保留候选柱数量的激活柱
                                    activeState(end+1:end+candidateNum) = columns(earlyIndex(currentcolumns));%加入激活柱
                                    break;
                                end
                            end
                            candidateNum = 0; %激活柱选择完毕
                        else %若激活数目低于候选柱的数量
                            activeState(end+1:end+activeLen) = columns; %全部加入候选柱
                            candidateNum = candidateNum - activeLen;    %更新候选柱的数目
                        end
                    end
                    %判断激活柱是否挑选完毕，==0，没有完成，证明剩下的都是未点火的，选择膜电压最大的那几个
                    if (candidateNum ~= 0)          %激活柱还剩有柱未选择
                        AllV(:,activeState) = -Inf; %剔除激活柱的膜电压，它们肯定是最大的
                        maxV=max(AllV);             %计算每个柱的最大膜电压
                        [maxV,columns] = sort(maxV,'descend');
                        if maxV(1) == maxV(end)
                            fprintf('cant choose the column!!!,the column is not enough\n');
                        else
                            activeState(end+1:end+candidateNum) = columns(1:candidateNum);
                        end
                    end
                end
            end
            %% 当前激活柱选择完毕，新选择的激活柱的激活数加1，并存储激活的柱和输出
            countActiveColumn(activeState) = countActiveColumn(activeState) +1;
            activeState = sort(activeState);        %激活柱的index按从小到大排序
            activeColumnAll(word,:) = activeState;  %存储当前激活柱
            activeTout = toutColumn(activeState,:); %获取激活柱的输出
            activeToutAll(word,:) = activeTout;     %存储当前激活柱的输出
            if (any(activeTout == Inf))             %是否存在没有输出的柱
                hasNoOutNum = hasNoOutNum + 1 ;     %个数加1 
            end
            %% STDP learn 更新输入到激活柱的连接（前馈连接）
            for activeIndex = 1:activeNum          %遍历每个激活柱
                activeColumn = activeState(activeIndex);
                fireTime = toutColumn(activeColumn);%获取每个柱的激活时间
                fireTime(fireTime == Inf) = [];     %去掉多余的时间点
                if(isempty(fireTime))
                    fireTime = Tmax;                            %没有输出，把输出设为最大的点火时间
                end
                %% STDP 有很多重复的计算时间，后面再优化
                for j = 1:length(fireTime)
                    %% 输入的时间在点火时间之后，降低权重
                    tempMinus = fireTime(j) - input;
                    tempMinus(tempMinus>=0) = -Inf;%输入的时间在点火时间之前的设-inf，不降低
                    deltaWMinus = -learnRateNeg*sum(exp(tempMinus/tau),2);
                    %% 输入的时间在点火时间之前，增加权重
                    tempPlus = fireTime(j) - input;
                    tempPlus(tempPlus == -Inf) = Inf;%输入的时间在点火时间之后的设inf，不增加权重
                    tempPlus(tempPlus<=0) = Inf;
                    deltaPlus =  learnRatePos*sum(exp(-tempPlus/tau),2);
                    if activeTout(activeIndex) == Inf
                        deltaPlus = deltaPlus*1.2;
                    end
                    %% 权重该变量为增加和降低的总和
                    deltaW = deltaWMinus + deltaPlus;
                    inputWeight(:,activeColumn) = inputWeight(:,activeColumn)+deltaW;
                end
            end
            %% 抑制其他点火了的权重
            fireColumns = find(toutColumnNum == 1);
            inhColumns = setdiff(fireColumns,activeIndex);
            for j = 1:length(inhColumns)
                inhColumn = inhColumns(j);
                %% 输入的时间在点火时间之后，降低权重
                fireTime = T_step;
                tempMinus = fireTime - input;
                tempMinus(tempMinus>=0) = -Inf;%输入的时间在点火时间之前的设-inf，不降低
                deltaWMinus = -learnRateNeg*sum(exp(tempMinus/tau),2);
                %% 权重该变量为增加和降低的总和
                inputWeight(:,inhColumn) = inputWeight(:,inhColumn)+deltaWMinus;
            end
            
            %% 限制权重的最大值/最小检查权重是否高于最大值或者低于最小值
            inputWeight(inputWeight>maxWeight) = maxWeight;
            inputWeight(inputWeight<minWeight) = minWeight;  
        end
        %% 计算这一轮和上一轮该单词选择的柱之间的相似度
        chooseSimility = zeros(wordNum,1);
        for i = 1:wordNum                  %遍历所有单词
            activei = activeColumnAll(i,:);%获取该单词的激活柱
            if epoch~=1
                oldActivei = lastActiveColumn(i,:);%获取该单词上一时刻的激活柱
                %以下三行计算相似度
                diffNeurons = activei-oldActivei';
                [~,comIdx] = find(diffNeurons==0);
                chooseSimility(i) = length(comIdx)/activeNum;
            end
        end
        meanSimility = sum(chooseSimility(:))/wordNum;
        fprintf('epoch%d,和上一次选择的最小相似度:%f, 平均相似度: %f\n',epoch,min(chooseSimility(:)),meanSimility);
        fprintf('active column without output:%d\n',hasNoOutNum);
        %% 结束的遍历条件
        if meanSimility>0.99 %平均相似度为0.99
            break;
        end
    end
    %% 再次检查两个单词之间的相似度是否正确
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
    %% 保存好每个单词的柱的表示，以及每个柱的输出
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


