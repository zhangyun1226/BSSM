clear;
close all;
%% 时间池代码
%% 加载输入数据
load('data.mat');                                      
load('spation512.mat');
%% 初始化参数
seqenceNum = 1000;                                     %序列个数
initParameter;                                         %默认的参数
wordLen = size(sequence,2);                            %每个序列的最大长度
DelayTime = Tau_m*Tau_s*log(Tau_m/Tau_s)/(Tau_m-Tau_s);%延迟到达最大PSP的时间
targetLen = [];                                        %和目标神经元相连的神经元个数
targetThreshold = 1;                                   %目标神经元的激活阈值
targetWordNum = 4;                                     %目标单词4个可以使目标神经元膜电压到达1
wi = targetThreshold/(activeNum*targetWordNum);        %每个激活神经元到目标神经元的权重
segmentConnect = round(activeNum*0.7);                 %每个segment和激活神经元相连的个数-16
maxSegmentConnect = 4*segmentConnect;                  %每个segments最多记忆4个上文，且不要共用
segmentThreshold = segmentConnect-1;                   %预测时连接数超过这个阈值才计算膜电压
segmentWi = 1/(maxSegmentConnect);                     %segment中每根连接的权重
maxSegmentList = [10,20,30,40];                        %远端树突的数量
trail = 1;                                             %实验次数
for disSegNum = 1:length(maxSegmentList)               %遍历远端树突的数量
    maxSegment = maxSegmentList(disSegNum);            %提供的输入的个数
    t1=clock;
    for trailNum = 1:trail                            
        %% 每次实验参数重新初始化
        %神经元segments的位置
        for i = 1:columnNum
            for j = 1:neuronNum
                cellSegmentsIdx{i,j} = {};     
            end
        end
        contextActiveList = cell(columnNum,neuronNum); %存储第i柱的j神经元的下文segments，预测时使用
        targetActiveList = cell(columnNum,neuronNum);  %存储第i柱的j神经元的目标神经元，预测时使用
        segmentsList = {};                             %存储segments的信息，以及它是哪个神经元的segments
        targetsCell = {};                              %目标神经元，包含具体的神经元，权重以及代表的句子
        cellSegmentsNum = zeros(columnNum,neuronNum);  %每个神经元segments的数量,代表了其有多少个上文
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
                        %% 将其连接的目标神经元存储起来
                        targetInfo = targetActiveList(learnNeurons);
                        for preIdx = 1: length(targetInfo)
                            if isempty(find(targetInfo{preIdx} == seq))
                                targetInfo{preIdx} = [targetInfo{preIdx},seq];
                            end
                        end
                        targetActiveList(learnNeurons) = targetInfo;
                    end
                    %% 存储信息 targetsCell
                    targetsCell{seq,1} = targetWeights;                 %权重
                    targetsCell{seq,2} = targetDelays;                  %延迟
                    targetsCell{seq,3} = sequence(seq,1:sentenceLen);   %序列
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
                            currentSegNum = cellSegmentsNum(currentColumn,:); %获取当前激活柱的所有神经元的segments数目
                            minSegNum = min(currentSegNum);               %最少的segment数目
                            neurons = find(currentSegNum == minSegNum);   %选择具有最少segments的神经元们
                            chooseNeuron = neurons(unidrnd(length(neurons)));%从这些神经元中，随机选择一个神经元
                            %% step2.3遍历该神经元的segments，找到一个segment与连接对象没有交集，把这些新连接建立在上面
                            activeSegmentIdx = [];
                            segmentFlag = 0; %是否需要新建segments，0需要，1不需要
                            if minSegNum~=0  %如果选择的学习神经元的segment数量不为0
                                segmentsIdx = cellSegmentsIdx{currentColumn,chooseNeuron};%获取该神经元的segments
                                for segI = 1:minSegNum           %遍历segments
                                    segment = segmentsList{segmentsIdx{segI},1};   % segmenti的信息
                                    segmentSynIdx = segment(:,1);                % segmenti连接的神经元的index
                                    if length(segmentSynIdx)~=maxSegmentConnect  % 如果当前segment的连接个数没有满最大值
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
                                            segmentsList{segmentsIdx{segI},1} = segment;                      %更新segment信息
                                            segmentFlag =1;                                                   %连接已经添加，无需新建
                                            activeSegmentIdx = segmentsIdx{segI};                             %连接添加的segment的位置
                                            break;
                                        end
                                    end
                                end
                            end
                            
                            %% step2.4 没有找到合适的segment添加信息，新建一个segments给该神经元
                            if segmentFlag == 0&&minSegNum<maxSegment
                                %% 新建的segments
                                segment = zeros(segmentConnect,4);
                                segment(:,1) = chooseIndex;       %2.4.1：segments的上文的神经元的index
                                segment(:,2) = segmentWi;         %2.4.2：segments的上文的神经元的权重
                                segment(:,3) = sample;            %2.4.3：segments的标签
                                %% 2.4.4 delay建立
                                postTout = currentTout(column);            %当前柱的输出
                                preTouts = seqActiveTout{word-1};          %取上一秒的激活柱的输出
                                preTouts = preTouts(connectedColumn);      %选择的柱的输出
                                delay = postTout - (preTouts + DelayTime); %计算延迟
                                segment(:,4) = delay;                      %2.4.4存储到segments的上文的神经元的延迟
                                %% step3: 将新建的segment加入segment总列表
                                segmentNum = sum(cellSegmentsNum(:)) + 1;             %新segment的idx
                                segmentsList{segmentNum,1} = segment;                 %加入总列表
                                segmentsList{segmentNum,2} = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron); %填写所对应的神经元
                                % 该神经元的segment的位置
                                cellSegmentsIdx{currentColumn,chooseNeuron} = [cellSegmentsIdx{currentColumn,chooseNeuron},segmentNum];
                                cellSegmentsNum(currentColumn,chooseNeuron) = cellSegmentsNum(currentColumn,chooseNeuron)+1;   %该神经元的segment数加1
                                neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1; %选择神经元的连接数增加1
                                activeSegmentIdx = segmentNum;                     %新segment位置
                            end
                            LearnState(currentColumn,chooseNeuron,word) = 1;       %学习神经元状态为1
                            
                            %% step2.5 将激活神经元连接的下文context存储起来
                            contextInfo = contextActiveList(chooseIndex);
                            for preIdx = 1: length(contextInfo)
                                contextInfo{preIdx} = [contextInfo{preIdx},activeSegmentIdx];
                            end
                            contextActiveList(chooseIndex) = contextInfo;
                        end
                    end
                end
            end
        end
        
        fprintf('learning is over!!!\n');
        fprintf('testing star!!!\n');
        
        %% test 只输入部分单词
        predictNoneThreshold = 0.1; %膜电压阈值，若低于，不让其预测出结果
        predictHasThreshold = 0.5;  %膜电压阈值，若高于，其预测出结果可以多个
        diffThresholdMany = 0.005;
        diffThresholdNone = 0.0005;
        endTime = DelayTime+T_step;
        segmentsLen = length(segmentsList);
        zerosInit = zeros(columnNum,neuronNum);
        minTargetThreshold = 0;
        InfInit = Inf*ones(columnNum,neuronNum);
        %% 测试序列的生成
        testInputNum = 3;                                      %输入单词的个数
        testSeq = sequence;                %测试序列等于输入序列
        testSeq(:,testInputNum+1:end) = -1;%测试序列输入个数以后的清空
        seqenceNum = size(sequence,1);     %输入序列的总句数
        testSeqNum = seqenceNum;           %测试序列的总句数
        
        %% 去除重复的输入
        for i = 1:seqenceNum          %遍历所有句子
            if i>testSeqNum
                break;                %如果重复测试的句子被删除后，句数已经到末尾了
            end
            currentSeq = testSeq(i,1:testInputNum); %获取当前第i个句子的序列
            delSeqIdx = [];                         %用于存放是否有和它一样的句子，这样的句子需要被删除
            for j = i+1:testSeqNum                  %查看当前句子以后的所有测试句子
                compareSeq = testSeq(j,1:testInputNum);%获取句子j
                if all(currentSeq - compareSeq == 0)   %若两个句子一摸一样
                    delSeqIdx(end+1) = j;              %j句子加入删除序列
                end
            end
            if length(delSeqIdx)~=0                          %若有句子在删除序列中
                testSeqNum = testSeqNum - length(delSeqIdx); %测试句子总数下降
                testSeq(delSeqIdx,:) = [];                   %去掉删除序列的句子
            end
        end
        wordLen = size(testSeq,2);    %测试句子的最大单词个数
        
        %% 保存位置以及信息
        str = 'test_segmentNum_';
        str = [str,num2str(maxSegment)];
        str = [str,'_'];
        str = [str,num2str(trailNum)];
        str = [str,'.txt'];
        fileID = fopen(str,'w');
        
        %% 存储第一个单词的信息
        FirstWordInfo = {};
        for sentencei = 1:max(testSeq(:))
            FirstWordInfo{sentencei,1} = {};
        end
        
        %% 测试开始
        for seq = 1:testSeqNum
            %             seq
            predictNeuronsState = zeros(columnNum,neuronNum,wordLen);%0:nT的预测状态
            for word = 1:wordLen                       %遍历当前句子的有输入的单词
                sample = testSeq(seq,word);            %获取具体的单词的样本
                if sample ~= -1                        %句子未结束
                    %% A.激活阶段
                    %% step1: 获取当前的激活柱和每个激活柱的输出
                    activecolumns = columnCell(sample,:);      %当前的激活柱
                    startT = (word-1)*Tmax;                    %属于哪个周期
                    currentTout = columnOut(sample,:) + startT;%激活柱的输出
                    
                    %% step2: 确定激活柱里的激活神经元，柱里要么激活所有，要么激活预测状态的神经元
                    activeNeurons = zerosInit;   %当前周期的激活神经元,全部初始化为0
                    activeTout = InfInit;        %当前周期的激活神经元的输出,全部初始化为Inf
                    if (word==1)
                        %% step2.1 第一个单词是没有预测神经元的，全激活
                        activeNeurons(activecolumns,:) = 1;%该柱所有神经元都处于激活状态
                        activeTout(activecolumns,:) = repmat(currentTout',1,neuronNum);
                    else
                        %% step2.1 其他单词，依次检测每个激活柱，查看其是否存在预测神经元
                        for column = 1:activeNum           %遍历激活柱
                            flagPredect = 0;               %当前柱是否有神经元被选为预测神经元，0无1有
                            currentColumn = activecolumns(column); %当前激活柱
                            preNeuron = find(predictNeuronsState(currentColumn,:,word-1)==1); %查看当前柱的哪些神经元处于预测状态
                            
                            %% step 2.2 存在预测神经元，激活所有预测神经元
                            if ~isempty(preNeuron)
                                flagPredect = 1;                           %存在预测神经元
                                activeNeurons(currentColumn,preNeuron) = 1;%该神经元变为激活神经元
                                activeTout(currentColumn,preNeuron) = currentTout(column);
                            end
                            
                            %% step 2.3 不存在预测神经元，激活整个柱
                            if (flagPredect == 0)
                                activeNeurons(currentColumn,:) = 1;
                                activeTout(currentColumn,:) =  currentTout(column);
                            end
                        end
                    end
                    
                    %% 激活的神经元对应的输出
                    activeIndex = find(activeNeurons == 1);                %激活神经元所对应的index
                    actTout = activeTout(activeIndex);
                    %% 将激活神经元/输出/以及样本保存到inputInfo中
                    if word == 1
                        inputInfo{1} = sample;
                        inputInfo{2} = activeIndex;
                        inputInfo{3} = actTout;
                    else
                        inputInfo{1} = [inputInfo{1};sample];
                        inputInfo{2} = [inputInfo{2};activeIndex];
                        inputInfo{3} = [inputInfo{3};actTout];
                    end
                    
                    %% B.预测阶段,用激活神经元做预测
                    if word == 1 && ~isempty(FirstWordInfo{sample})  %若已经学习过，那么信息已经存储下来了，直接调用就行
                        predictState = FirstWordInfo{sample,1};      %预测状态
                        %                         predictWordList = FirstWordInfo{sample,2};   %预测的单词列表
                        %                         predictInformation = FirstWordInfo{sample,3};%预测的神经元以及它的输出
                    else                                             %若不是第一个单词，或者第一个单词第一次出现，学习预测
                        segmentState = zeros(segmentsLen,1);   %segments与激活神经元的连接数量
                        %% step1:遍历所有激活神经元,查看哪些segments激活
                        for neuronIdx = 1:length(activeIndex)
                            neuronSegIndex = activeIndex(neuronIdx);        %激活的神经元
                            contextInfo = contextActiveList{neuronSegIndex};%获取该激活神经元的下文神经元和segments
                            for segI = 1:neuronConnectNum(neuronSegIndex)   %遍历所有下文神经元的激活segments
                                activeSegId = contextInfo(segI);            %获取segment的位置
                                segmentState(activeSegId) = segmentState(activeSegId) + 1;%该segment与激活神经元的连接数量加1
                            end
                        end
                        predictState = zeros(columnNum,neuronNum);   %每个神经元的预测状态，初始为0，无预测
                        activeSegments = find(segmentState >= segmentThreshold);%与激活神经元相连，连接数大于阈值的激活
                        for segIdx = 1:length(activeSegments)
                            actSegIdx = activeSegments(segIdx);      %获取激活的segments的位置
                            neuronSegIndex = segmentsList{actSegIdx,2};    %获取激活的segment所对应的神经元
                            predictState(neuronSegIndex) = 1;              %该神经元当前时刻变为预测状态
                        end
                        %% 存储第一个单词的信息
                        if word == 1
                            FirstWordInfo{sample,1} = predictState;
                            %                             FirstWordInfo{sample,2} = predictWordList;
                            %                             FirstWordInfo{sample,3} = predictInformation;
                        end
                    end
                    
                    %% 存储预测信息
                    predictNeuronsState(:,:,word) = predictState;
                else
                    outputInfo = {};                                %存储输出序列的信息
                    predictToutQuick;
                    hasContextNeurons = find(hasContextNeurons~=0); %有预测的激活神经元
                    %有预测的激活神经元《〈激活神经元，且不是全激活，那么该单词是叶子节点
                    if (length(activeIndex) - length(hasContextNeurons) >= segmentThreshold)&&length(activeIndex)~=activeNum*neuronNum
                        predicSegLen = size(outputInfo,1);
                        outputInfo{predicSegLen+1,1} = inputInfo{1};%第1列存该句子所有单词标签
                        outputInfo{predicSegLen+1,2} = inputInfo{2};%第2列存该句子所有激活神经元
                        outputInfo{predicSegLen+1,3} = inputInfo{3};%第3列存该所有激活的神经元的输出
                    end
                    %% 开始建树预测-不同树表示不同子句，单词表示不同节点
                    haveReadList = zeros(size(predictWordList));   %用于存储节点单词是否遍历过，初始化都未遍历过
                    while (~isempty(predictWordList))              %已经遍历完了，栈内没有任何元素了
                        %% 查找未遍历过的单词-子结点
                        listLen = find(haveReadList == 0);
                        %% 栈顶部不存在，删除信息退出
                        if(isempty(listLen)) || length(outputInfo)>seqenceNum
                            predictWordList(listLen) = [];
                            predictInformation(listLen,:) = [];
                            haveReadList(listLen) = [];
                            break;
                        end
                        listLen = listLen(end);                               %最上面那个结点-栈顶结点
                        currentWord = predictWordList(listLen);               %它所在的单词标签
                        currentSegmentsCell = predictInformation(listLen,1:2);%它所在的单词的预测神经元信息
                        haveReadList(listLen) = 1;                            %该单词已经用过了
                        %% 没有输入预测状态变为活跃状态
                        actIndex = currentSegmentsCell{1,2};               %激活神经元所在的index
                        if length(actIndex) < segmentThreshold             %若激活的神经元小于阈值，则不能代表一个完整的单词，去掉
                            %% 去掉这个单词，因为它是噪声，可能会导致栈顶不存在
                            predictWordList(listLen) = [];
                            predictInformation(listLen,:) = [];
                            haveReadList(listLen) = [];
                            continue;
                        end
                        
                        PredictToPredict;
                        %% step3 判断是否为叶子结点叶子结点1：判断当前是否有激活的神经元没有预测的，表明句子结束，加入列表
                        leaf = 0;
                        hasContextNeurons = find(hasContextNeurons~=0);                %有预测的激活神经元
%                         %预测的激活神经元<当前的激活神经元，且不是全激活的单词导致的
%                         if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold) &&length(actIndex)~=activeNum*neuronNum
%                             leaf = 1;
%                         end
                        %% 判断当前有没有新的预测，若有，那么加入栈，若没有，那么是叶子结点
                        nextLen = length(tempWordList);
                        if nextLen >= segmentThreshold %%非叶子结点
                            predictWordList(end+1:end+nextLen) = tempWordList;        %保存预测单词列表
                            predictInformation(end+1:end+nextLen,:) = tempInformation;%保存预测单词信息
                            haveReadList(end+1:end+nextLen) = 0;                      %这些结点未被遍历过
                        else
                            leaf = 1; %% 叶子结点
                        end
                        %% 叶子结点，获取子句，保留信息
                        if leaf == 1
                            %% 保存所有词的标签
                            wordList = find(haveReadList == 1);  %所有为1的单词组成这个句子
                            predicSegLen = size(outputInfo,1);   %当前存储子句的个数，将新子句加入末尾
                            temp = predictWordList(wordList);
                            if size(temp,2)>1
                                temp = reshape(temp,length(temp),1);
                            end
                            outputInfo{predicSegLen+1,1} = [inputInfo{1};temp];     %第1列存该句子所有单词标签
                            tout = [];
                            activeNeurons = [];
                            for preLen = 1:length(temp)
                                tout = [tout;predictInformation{wordList(preLen),2}];
                                activeNeurons = [activeNeurons;predictInformation{wordList(preLen),1}'];
                            end
                            outputInfo{predicSegLen+1,2} = [inputInfo{2};tout];%第2列存该句子所有单词激活的神经元
                            outputInfo{predicSegLen+1,3} = [inputInfo{3};activeNeurons];%第3列激活的神经元的输出
                            
%                         end
%                         if leaf == 1
                        %% 删除该子句句子相关信息
                            start = find(haveReadList == 0);    %找到从还没开始遍历的结点
                            
                            if isempty(start)                   %如果所有单词都已经遍历过，清空信息
                                predictWordList = [];
                                predictInformation = {};
                                haveReadList = [];
                            else                                  %存在未遍历的单词
                                start = start(end);               %找到最后一个未遍历的单词，清空它之后的所有信息
                                predictWordList(start+1:end) = [];
                                predictInformation(start+1:end,:) = [];
                                haveReadList(start+1:end) = [];
                            end
                        end
                        
                    end
                    %%
                    testTree;
                    
                    break;
                end
                
            end
        end
        fprintf('mean accuracy:%f\n',mean(seqAcc(1:seq,trailNum)));
        fprintf(fileID,'mean accuracy:%f\n',mean(seqAcc(1:seq,trailNum)));
    end
    
    t2=clock;
    runtime(disSegNum) = etime(t2,t1);
    fprintf('\n run-time:%f\n\n',runtime(disSegNum));
    fprintf('the mean accuracy of %d trails:%f\n',trail,mean(mean(seqAcc(1:testSeqNum,:))));
    fprintf(fileID,'the mean accuracy of %d trails:%f\n',trail,mean(mean(seqAcc(1:testSeqNum,:))));
    str(end-5:end)=[];
    save([str,'.mat'],'seqAcc');
end

