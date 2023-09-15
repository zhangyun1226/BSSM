clear;
close all;
%% 时间池代码
load('data.mat');                                      %加载输入数据
load('spation512.mat');
initParameter;                                         %初始化参数
seqenceNum = 1000;
wordLen = size(sequence,2);                            %每个序列的最大长度
DelayTime = Tau_m*Tau_s*log(Tau_m/Tau_s)/(Tau_m-Tau_s);%延迟到达最大PSP的时间
targetLen = [];                                        %和目标神经元相连的神经元个数
targetThreshold = 1;                                   %目标神经元的激活阈值
targetWordNum = 4;                                     %目标单词4个可以使目标神经元膜电压到达1
wi = targetThreshold/(activeNum*targetWordNum);        %每个激活神经元到目标神经元的权重
segmentConnect = round(activeNum*0.7);           %每个segment和激活神经元相连的个数-16
maxSegmentConnect = 4*segmentConnect;            %每个segments最多记忆4个上文，且不要共用
segmentThreshold = segmentConnect-1;             %预测时连接数超过这个阈值才计算膜电压
segmentWi = 1/(segmentThreshold);                %segment中每根连接的权重
%% 参数初始化
for i = 1:columnNum
    for j = 1:neuronNum
        segmentsList{i,j} = {};%存储第i柱的j神经元的权重
        delayList{i,j} = {};   %存储第i柱的j神经元的延迟
    end
end
targetsCell = {};                              %目标神经元，包含具体的神经元，权重以及代表的句子
segmentsNum = zeros(columnNum,neuronNum);      %每个神经元segments的数量,代表了其有多少个上文
neuronConnectNum = zeros(columnNum,neuronNum); %每个神经元向外连接的次数
%% 时间池学习
for seq = 1:1:seqenceNum                            %遍历所有句子序列
    LearnState = zeros(columnNum,neuronNum,wordLen);%0:nT的学习状态
    seqActiveTout = {};                             %存储0:nT所有输出
    for word = 1:wordLen                            %遍历当前句子序列的每个单词
        sample = sequence(seq,word);                %当前单词
        if sample == 0
            %% 序列结束
            %% 设置目标簇的权重以及延迟
            targetWeights = zeros(columnNum,neuronNum);  %到目标神经元的权重
            targetDelays = Inf*ones(columnNum,neuronNum);%到目标神经元的延迟
            sentenceLen = word-1;                        %当前句子的长度
            for i = 1:sentenceLen                        %遍历句子序列的每个单词
                learnNeurons = find(LearnState(:,:,i) == 1); %该句子第i个单词的学习神经元
                targetWeights(learnNeurons) = wi;            %第i个单词的学习神经元到目标的权重
                targetDelays(learnNeurons) = (word - i)*Tmax;%第i个单词的激神经元到目标的延迟；nT,(n-1)T...T
            end
            %% 存储信息 targetsCell
            targetsCell{seq,1} = targetWeights;                 %权重
            targetsCell{seq,2} = targetDelays;                  %延迟
            targetsCell{seq,3} = sequence(seq,1:sentenceLen);   %序列
            %                     %% 存储每个句子被覆盖了多少
            %                     targetLenCell{seq,1} = length(find(targetWeights>0));
            %                     targetLenCell{seq,2} = sentenceLen*activeNum;
            %                     targetLenCell{seq,3} = targetLenCell{seq,1}-targetLenCell{seq,2};
            break;
        else
            %% one trail 学习句子
            %% A.激活阶段
            %% step1:获取当前的激活柱和每个激活柱的输出
            activecolumns = columnCell(sample,:);      %当前的激活柱
            startT = (word-1)*Tmax;                    %属于哪个周期
            currentTout = columnOut(sample,:) + startT;%激活柱的输出
            seqActiveTout{word} = currentTout;         %存储激活柱的输出
            %% step2: 确定激活柱里的激活神经元，柱里要么激活所有，要么激活预测状态的神经元
            activeNeurons = zeros(columnNum,neuronNum);%激活神经元
            if (word==1)                               %第一个单词选择向外连接数最少的作为激活神经元
                %% 选择向外连接数最少的作为学习神经元
                for num = 1:activeNum                                            %遍历激活柱
                    currentColumn = activecolumns(num);                          %激活柱i
                    neronsCon = neuronConnectNum(currentColumn,:);               %柱i的神经元向外的连接数目
                    minConNeuron = find(neronsCon == min(neronsCon));            %选择最少连接的数的神经元
                    currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %可能有多个，随机选择一个神经元
                    LearnState(currentColumn,currentNeuron,word)=1;              %将该神经元加入激活列表
                end
            else         %其他单词
                preLearnState = LearnState(:,:,word-1);                          %获取上一时刻的学习神经元
                %% 当前单词的学习
                for column = 1:activeNum                                         %遍历激活柱
                    %% step2.1:为segment选择要连接的神经元，随机选择segmentConnect的神经元作为segment的连接对象
                    [preLearnColumns,preLearnNeurons] = find(preLearnState == 1);%获取上一时刻学习神经元的位置
                    [preLearnColumns,columnIndex] = sort(preLearnColumns);       %把柱按序排列，一定需要，因为输出是按柱的升序排列的
                    preLearnNeurons = preLearnNeurons(columnIndex);              %相应的神经元排列
                    connectedColumn = sort(randperm(length(preLearnColumns),segmentConnect));%从所有学习神经元随机选择segmentConnect相连神经元
                    randActiveColumns = preLearnColumns(connectedColumn);        %随机选择的激活神经元所处的柱
                    randActiveNeurons = preLearnNeurons(connectedColumn);        %随机选择的激活神经元所处的柱内序号
                    chooseIndex = sub2ind(size(neuronConnectNum),randActiveColumns,randActiveNeurons); %将这些选择神经元转换为index
                    %% step2.2：选择当前单词的激活柱中具有最少segments的神经元
                    currentColumn = activecolumns(column);        %当前的激活柱i
                    currentSegNum = segmentsNum(currentColumn,:); %获取当前激活柱的所有神经元的segments数目
                    minSegNum = min(currentSegNum);               %最少的segment数目
                    neurons = find(currentSegNum == minSegNum);   %选择具有最少segments的神经元们
                    chooseNeuron = neurons(unidrnd(length(neurons)));%从这些神经元中，随机选择一个神经元
                    %% step2.3 遍历该神经元的segments，找到一个segment与连接对象没有交集，把这些segments建立在上面
                    segmentFlag = 0; %是否需要新建segments，0需要，1不需要
                    if minSegNum~=0  %如果选择的学习神经元的segment数量不为0
                        segments = segmentsList{currentColumn,chooseNeuron};%获取该神经元的segments
                        for segIndex = 1:minSegNum           %遍历segments
                            segment = segments{segIndex};    %segmenti的信息
                            segmentSynIdx = segment(:,1);    %segmenti连接的神经元的index
                            if length(segmentSynIdx)~=maxSegmentConnect  %如果当前segment的连接个数没有满最大值
                                
                                diffNeurons = chooseIndex-segmentSynIdx';     %segments连接的神经元和现在选择的连接对象之差
                                [~,comIdx] = find(diffNeurons==0);            %为0表示有相同的元素
                                if isempty(comIdx)                            %没有交集
                                    %% segment加入当前的信息
                                    segmentLen = length(segmentSynIdx);       %当前segment包含的神经元个数
                                    segment( segmentLen+1: segmentLen+segmentConnect,1) = chooseIndex;%2.3.1segments的上文的神经元的index
                                    segment( segmentLen+1: segmentLen+segmentConnect,2) = segmentWi;  %2.3.2到segments的上文的神经元的权重
                                    segment( segmentLen+1: segmentLen+segmentConnect,3) = sample;     %2.3.3当前单词的信息
                                    %% delay
                                    postTout = currentTout(column);           %当前柱的输出
                                    preTouts = seqActiveTout{word-1};         %取上一秒的激活柱的输出
                                    preTouts = preTouts(connectedColumn);     %选择的柱的输出
                                    delay = postTout - (preTouts + DelayTime);%计算延迟
                                    segment( segmentLen+1: segmentLen+segmentConnect,4) = delay;      %2.3.4存储到segments的上文的神经元的延迟
                                    %%
                                    neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1; %选择神经元的连接数增加1
                                    segments{segIndex} =  segment;                                    %更新segment信息
                                    segmentsList{currentColumn,chooseNeuron} = segments;              %segment加入列表
                                    segmentFlag =1;
                                    break;
                                end
                            end
                        end
                    end
                    %% step2.4 没有找到合适的segment添加信息，新建一个segments给该神经元
                    if segmentFlag == 0
                        %% 新建的segments
                        segment = zeros(segmentConnect,4);
                        segment(:,1) = chooseIndex;       %2.4.1：segments的上文的神经元的index
                        segment(:,2) = segmentWi;         %2.4.2：segments的上文的神经元的权重
                        segment(:,3) = sample;            %2.4.3：segments的标签
                        %% 2.4.4 delay建立
                        postTout = currentTout(column);   %当前柱的输出
                        preTouts = seqActiveTout{word-1}; %取上一秒的激活柱的输出
                        preTouts = preTouts(connectedColumn);      %选择的柱的输出
                        delay = postTout - (preTouts + DelayTime); %计算延迟
                        segment(:,4) = delay;                      %2.4.4存储到segments的上文的神经元的延迟
                        %% step3: 将新建的segment加入segment列表
                        num = segmentsNum(currentColumn,chooseNeuron);            %当前学习神经元拥有的segments的数量
                        currentSegCell = segmentsList{currentColumn,chooseNeuron};%当前学习神经元拥有的segments
                        currentSegCell{num+1} = segment;                          %将新segment写入
                        segmentsList{currentColumn,chooseNeuron} = currentSegCell;%更新信息
                        neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1; %选择神经元的连接数增加1
                        segmentsNum(currentColumn,chooseNeuron) = num+1;                  %该神经元的segment数加1
                    end
                    LearnState(currentColumn,chooseNeuron,word) = 1;       %学习神经元状态为1
                end
            end
        end
    end
end

fprintf('learning is over!!!\n');
fprintf('testing star!!!\n');


