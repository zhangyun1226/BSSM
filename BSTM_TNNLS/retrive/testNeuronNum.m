clear;
close all;
%% 时间池代码
load spation512.mat
load data.mat
initParameter;        %初始化参数
neuronNumList = [4,8,16,32,64,128];   %% 每个柱的神经元的个数
maxPredicSen = 50;
trail = 2;
%% 存储正确率
seqenceNum = 1000;
for neuNum = 1:1%length(neuronNumList)
    neuronNum = neuronNumList(neuNum);
    segmentConnect = round(activeNum*0.7);  %每个segment和激活神经元相连的个数-16
    maxSegmentConnect = 4*segmentConnect; %每个segments最多记忆8个上文，且不要共用
    segmentThreshold = segmentConnect-1;%预测时超过15个，计算膜电压，便于计算
    segmentWi = 1/(segmentConnect); %segment中每根连接的权重
    seqAcc = zeros(seqenceNum,trail);
    for testN = 1:trail
        %% 参数初始化
        for num = 1:columnNum
            for j = 1:neuronNum
                segmentsList{num,j} = {};%存储权重
                delayList{num,j} = {};   %存储延迟
            end
        end
        targetsCell = {};                                      %目标神经元，包含具体的神经元，权重以及代表的句子
        segmentsNum = zeros(columnNum,neuronNum);              %每个神经元有几个segment,代表了其有多少个上文
        neuronConnectNum = zeros(columnNum,neuronNum);         %每个神经元向外连接的次数，除以16/20就是下文的个数
        
        sequence = sequence(1:seqenceNum,:);
        wordLen = size(sequence,2);                            %每个序列的最大长度
        DelayTime = Tau_m*Tau_s*log(Tau_m/Tau_s)/(Tau_m-Tau_s);%延迟到达最大PSP的时间
        targetLen = [];                                        %和目标神经元相连的神经元个数
        targetThreshold = 1;                                   %目标神经元的激活阈值
        %% 时间池学习
        commonlength =[];                                  %存储每次预测时激活的segment与激活柱的公共神经元数量
        actIndexLen = [];                                  %存储每次激活的神经元个数
        for seq = 1:1:seqenceNum                           %遍历训练序列
            LearnState = zeros(columnNum,neuronNum,wordLen);%0:nT的学习状态
            seqActiveColumns = zeros(activeNum,wordLen);   %存储0:nT所有激活柱
            seqActiveTout = {};                            %存储0:nT所有输出
            for word = 1:wordLen                           %遍历每个单词
                sample = sequence(seq,word);               %当前样本
                if sample == 0 %% 序列结束
                    %% 设置目标簇的权重以及延迟
                    targetWeights = zeros(columnNum,neuronNum); %非学习神经元对该目标是抑制
                    targetDelays = Inf*ones(columnNum,neuronNum);
                    sentenceLen = word-1;
                    targetWordNum = 4;
                    wi = 1/(activeNum*targetWordNum);
                    for i = 1:sentenceLen
                        learnNeurons = find(LearnState(:,:,i) == 1); %该句子第i个单词的学习神经元
                        targetWeights(learnNeurons) = wi;            %第i个单词的学习神经元到目标的权重
                        targetDelays(learnNeurons) = (word - i)*Tmax;%第i个单词的激神经元到目标的延迟
                    end
                    targetsCell{seq,1} = targetWeights;
                    targetsCell{seq,2} = targetDelays;
                    targetsCell{seq,3} = sequence(seq,1:sentenceLen);
                    targetLenCell{seq,1} = length(find(targetWeights>0));
                    targetLenCell{seq,2} = sentenceLen*activeNum;
                    targetLenCell{seq,3} = targetLenCell{seq,1}-targetLenCell{seq,2};
                    break;
                else %% one trail 学习句子
                    %             fprintf('第%d句，第%d个单词,%s\n',seq,word,inputLablesCell{sample});
                    %             fprintf(fileID,'第%d句，第%d个单词,%s\n',seq,word,inputLablesCell{sample});
                    %% A.激活阶段
                    %% step1:获取当前的激活柱和每个激活柱的输出
                    activecolumns = columnCell(sample,:);      %当前的激活柱
                    seqActiveColumns(:,word) = activecolumns;%存储
                    startT = (word-1)*Tmax;                  %属于哪个周期
                    currentTout = columnOut(sample,:) + startT;%激活柱的输出
                    seqActiveTout{word} = currentTout;       %存储激活柱的输出
                    
                    %% step2: 确定激活柱里的激活神经元，柱里要么激活所有，要么激活预测状态的神经元
                    activeNeurons = zeros(columnNum,neuronNum);
                    if (word==1)                             %第一个单词是没有上文的segments的，全激活
                        %% 选择向外连接数最少的作为学习神经元
                        for num = 1:activeNum                                 %遍历上一个周期激活的柱
                            currentColumn = activecolumns(num);               %上一周期的激活柱i
                            neronsCon = neuronConnectNum(currentColumn,:);    %柱i的神经元向外的连接数目
                            minConNeuron = find(neronsCon == min(neronsCon)); %选择最少连接的激活数的神经元
                            currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %可能有多个，随机选择一个神经元
                            LearnState(currentColumn,currentNeuron,word)=1;                       %将该神经元加入列表
                        end
                    else         %其他单词
                        preLearnState = LearnState(:,:,word-1);                  %获取上一时刻的学习神经元
                        %% 当前单词的学习
                        for column = 1:activeNum               %遍历激活柱
                            currentColumn = activecolumns(column);                 %当前的激活柱
                            %% 开始学习
                            %% step2.1:为segment选择要连接的神经元，随机选择百分之80的神经元作为segment的连接对象
                            [preLearnColumns,preLearnNeurons] = find(preLearnState == 1);%获取学习神经元的位置
                            [preLearnColumns,columnIndex] = sort(preLearnColumns);%把柱按序排列，一定需要，因为输出是按升序排列的
                            preLearnNeurons = preLearnNeurons(columnIndex);       %相应的神经元排列
                            connectedColumn = sort(randperm(length(preLearnColumns),segmentConnect));%从所有学习神经元随机选择segmentConnect相连神经元
                            randActiveColumns = preLearnColumns(connectedColumn);        %随机选择的激活神经元所处的柱
                            randActiveNeurons = preLearnNeurons(connectedColumn);        %随机选择的激活神经元所处的柱内序号
                            chooseIndex = sub2ind(size(neuronConnectNum),randActiveColumns,randActiveNeurons); %转换为index
                            %% step2.2：选择当前单词的激活柱中具有最少segments的神经元
                            currentSegNum = segmentsNum(currentColumn,:);             %获取当前激活柱的所有神经元的segments数目
                            minSegNum = min(currentSegNum);
                            neurons = find(currentSegNum == minSegNum);         %具有最少segments的神经元们
                            chooseNeuron = neurons(unidrnd(length(neurons)));      %从这些神经元中，随机选择一个神经元
                            %% step2.3 遍历该神经元的segments，找到一个segment与连接对象没有交集，把这些segments建立在上面
                            segmentFlag = 0;
                            if minSegNum~=0
                                segments = segmentsList{currentColumn,chooseNeuron};       %获取该神经元的segments
                                for segIndex = 1:minSegNum
                                    segment = segments{segIndex};    %当前segment的信息
                                    segmentSynIdx = segment(:,1);      %权重连接的神经元的index
                                    if length(segmentSynIdx)~=maxSegmentConnect  %如果当前segment的连接个数没有满最大值
                                        %segments连接的神经元和连接对象的是否有交集
                                        diffNeurons = chooseIndex-segmentSynIdx';
                                        diffNeurons(diffNeurons~=0) = 1;
                                        diffNeurons = sum(diffNeurons);
                                        if all(diffNeurons == segmentConnect) %没有交集
                                            %% 将信息加入当前的segment
                                            segmentLen = length(segmentSynIdx);
                                            segment( segmentLen+1: segmentLen+segmentConnect,1) = chooseIndex;
                                            segment( segmentLen+1: segmentLen+segmentConnect,2) = segmentWi;         %权重
                                            segment( segmentLen+1: segmentLen+segmentConnect,3) = sample;
                                            
                                            %% delay
                                            postTout = currentTout(column);               %当前柱的最早输出，最新只有一个输出
                                            preTouts = seqActiveTout{word-1};          %取上一秒的激活柱的最早输出
                                            preTouts = preTouts(connectedColumn);
                                            delay = postTout - (preTouts + DelayTime);      %计算延迟
                                            segment( segmentLen+1: segmentLen+segmentConnect,4) = delay;
                                            
                                            neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1;         %选择神经元的连接数增加1
                                            
                                            segments{segIndex} =  segment;
                                            segmentsList{currentColumn,chooseNeuron} = segments;
                                            segmentFlag =1;
                                            break;
                                        end
                                    end
                                end
                            end
                            %% step2.4 没有找到合适的segment添加信息，新建一个segments给该神经元
                            if segmentFlag == 0
                                segment = zeros(segmentConnect,2);%新建的segments
                                segment(:,1) = chooseIndex;       %segments的上文的神经元的index
                                segment(:,2) = segmentWi;         %segments的权重
                                segment(:,3) = sample;            %segments的标签
                                %% delay建立
                                postTout = currentTout(column);               %当前柱的最早输出，最新只有一个输出
                                preTouts = seqActiveTout{word-1};          %取上一秒的激活柱的最早输出
                                preTouts = preTouts(connectedColumn);
                                delay = postTout - (preTouts + DelayTime);      %计算延迟
                                segment(:,4) = delay;
                                %% step3: 将新建的segment加入segment列表
                                num = segmentsNum(currentColumn,chooseNeuron);
                                currentSegCell = segmentsList{currentColumn,chooseNeuron};
                                currentSegCell{num+1} = segment; %第一行为segment
                                segmentsList{currentColumn,chooseNeuron} = currentSegCell;
                                
                                neuronConnectNum(chooseIndex) = neuronConnectNum(chooseIndex) +1;         %选择神经元的连接数增加1
                                segmentsNum(currentColumn,chooseNeuron) = num+1; %该神经元的segment数加1
                            end
                            LearnState(currentColumn,chooseNeuron,word) = 1;       %柱中选择的具有最少segments的神经元被选为学习神经元
                        end
                    end
                end
            end
        end
        fprintf('learning is over!!!\n');
        fprintf('testing star!!!\n');
        
        %% test is star
        %% test 增加部分单词
        predictNoneThreshold = 0.1; %膜电压阈值，若低于，不让其预测出结果
        predictHasThreshold = 0.5;    %膜电压阈值，若高于，其预测出结果可以多个
        diffThresholdMany = 0.005;
        diffThresholdNone = 0.0005;
        endTime = DelayTime+T_step;
        %% 测试序列的生成
        %% 测试序列的生成
        testInputNum = 3;                  %提供的输入的个数
        testSeq = sequence;           %测试序列等于输入序列
        testSeq(:,testInputNum+1:end) = -1;%测试序列输入个数以后的清空
        seqenceNum = size(sequence,1); %输入序列的总句数
        testSeqNum = seqenceNum;      %测试序列的总句数
        
        %% 去除重复的输入
        for i = 1:seqenceNum          %遍历所有句子
            if i>testSeqNum
                break;                %如果重复测试的句子被删除后，句数已经到末尾了
            end
            currentSeq = testSeq(i,1:testInputNum); %获取当前第i个句子的序列
            delSeqIdx = [];                    %用于存放是否有和它一样的句子，这样的句子需要被删除
            for j = i+1:testSeqNum             %查看当前句子以后的所有测试句子
                compareSeq = testSeq(j,1:testInputNum);  %获取句子j
                if all(currentSeq - compareSeq == 0)%若两个句子一摸一样
                    delSeqIdx(end+1) = j;           %j句子加入删除序列
                end
            end
            if length(delSeqIdx)~=0            %若有句子在删除序列中
                testSeqNum = testSeqNum - length(delSeqIdx); %测试句子总数下降
                testSeq(delSeqIdx,:) = [];                   %删除该句子
            end
        end
        maxSenLen = size(testSeq,2);  %测试句子的单词个数 == testNum
        wordLen = size(testSeq,2);    %测试句子的单词个数 == testNum
        
        fprintf('Neuron number %d, trail %d.\n', neuronNum,testN);
        fprintf('Input words %d\n', testInputNum);
        
        %% 保存位置以及信息
        str = 'test_Neuron_';
        str = [str,num2str(neuronNum)];
        str = [str,'_'];
        str = [str,num2str(testN)];
        str = [str,'.txt'];
        fileID = fopen(str,'w');
        
        %% 存储第一个单词的信息
        FirstWordInfo = {};
        for i = 1:max(testSeq(:))
            FirstWordInfo{i,1} = {};
        end
        
        %% 测试开始
        for seq = 1:testSeqNum
            preState = zeros(columnNum,neuronNum,wordLen);%0:nT的预测状态
            LearnState = zeros(columnNum,neuronNum,wordLen);
            outputInfo = {};                           %存储序列的信息
            
            for word = 1:wordLen
                sample = testSeq(seq,word);
                if sample ~= -1
                    %% A.激活阶段
                    %% step1: 获取当前的激活柱和每个激活柱的输出
                    activecolumns = columnCell(sample,:);      %当前的激活柱
                    startT = (word-1)*Tmax;                  %属于哪个周期
                    currentTout = columnOut(sample,:) + startT;%激活柱的输出
                    
                    %% step2: 确定激活柱里的激活神经元，柱里要么激活所有，要么激活预测状态的神经元
                    newWordFlag = 0;
                    activeNeurons = zeros(columnNum,neuronNum);
                    if (word==1)
                        %% step2.1 第一个单词是没有预测神经元的，全激活
                        activeNeurons(activecolumns,:) = 1;%该柱所有神经元都处于激活状态
                        newWordFlag = 1;
                    else
                        %% step2.1 其他单词，依次检测每个激活柱，查看其是否存在预测神经元
                        preLearnState = LearnState(:,:,word-1);
                        for column = 1:activeNum           %遍历激活柱
                            flagPredect = 0;               %当前柱是否有神经元被选为预测神经元，0无1有
                            currentColumn = activecolumns(column); %当前激活柱
                            preNeuron = find(preState(currentColumn,:,word-1)==1); %查看当前柱的哪些神经元处于预测状态
                            
                            %% step 2.2 存在预测神经元，激活所有预测神经元
                            if length(preNeuron) >=1
                                flagPredect = 1;
                                activeNeurons(currentColumn,preNeuron) = 1;%该神经元变为激活神经元
                            end
                            
                            %% step 2.3 不存在预测神经元，激活整个柱，不存在这项
                            if (flagPredect == 0)
                                %该柱所有神经元都处于激活状态
                                activeNeurons(currentColumn,:) = 1;
                            end
                        end
                        
                        if all(activeNeurons(activecolumns,:) == 1)
                            newWordFlag = 1;
                        end
                        
                    end
                    
                    %% 激活的神经元对应的输出
                    actIndex = find(activeNeurons == 1);                %激活神经元所对应的index
                    actTout = zeros(size(actIndex));                    %每个激活神经元的输出
                    [columnIdx,~] = ind2sub(size(segmentsNum),actIndex);%激活神经元所对应的柱
                    for i = 1:activeNum
                        idx = find(columnIdx == activecolumns(i));      %哪些神经元的输出是激活柱i的
                        actTout(idx,:) = currentTout(i);              %柱i的输出赋值给它
                    end
                    %% 将激活神经元/输出/以及样本保存到trainInfo中
                    if word == 1
                        trainInfo{1} = sample;
                        trainInfo{2} = actIndex;
                        trainInfo{3} = actTout;
                    else
                        trainInfo{1} = [trainInfo{1},sample];
                        trainInfo{2} = [trainInfo{2};{actIndex}];
                        trainInfo{3} = [trainInfo{3};{actTout}];
                    end
                    
                    %% B.预测阶段，若存在激活神经元
                    if newWordFlag == 1 && ~isempty(FirstWordInfo{sample})
                        predictState = FirstWordInfo{sample,1};
                        predictWordList = FirstWordInfo{sample,2};
                        predictInformation = FirstWordInfo{sample,3};
                    else
                        predictState = zeros(columnNum,neuronNum);     %每个神经元的预测状态，初始为0，无预测
                        neuronHasSegmentsIndexs = find(segmentsNum>0); %获取有segments的神经元
                        predictWordList = [];                          %预测的单词列表
                        predictInformation = {};                       %预测的神经元以及它的输出
                        hasContextNeurons = zeros(columnNum,neuronNum);%每个预测神经元是由哪些神经元导致的，初始为0，没有导致下一个的预测
                        %% step1:遍历所有segment,得出预测神经元
                        for num = 1:length(neuronHasSegmentsIndexs)        %遍历所有有segments的神经元
                            neuronSegIndex = neuronHasSegmentsIndexs(num); %神经元的位置
                            segments = segmentsList{neuronSegIndex};       %获取该神经元的segments
                            
                            %% step2 判断该segments和激活神经元有相交的神经元没有，小于阈值，则不用计算膜电压了
                            for segIndex = 1:segmentsNum(neuronSegIndex) %该神经元segments的个数
                                
                                segmentW = segments{segIndex};    %当前segment的连接权重信息
                                %segmentlabel = segments{2,segIndex};%当前segment的标签
                                segmentSynIdx = segmentW(:,1);      %权重连接的神经元的index
                                
                                %% 比较segment有权重的神经元和激活神经元的交集
                                diffNeurons = actIndex-segmentSynIdx';
                                [~,comIdx] = find(diffNeurons==0);
                                
                                %% 判断该segments的交集是否大于阈值，若小于阈值，则不计算膜电压
                                if (length(comIdx) >= segmentThreshold)
                                    commonNeurons = segmentSynIdx(comIdx);
                                    %% step3 计算segment在下个周期的膜电压，膜电压最大的时候为其输出
                                    %% 3.1 取出公共神经元的延迟和权重
                                    delay = segmentW(comIdx,4);              %该神经元对应的segment的延迟
                                    weight = segmentW(comIdx,2);             %segment连接的权重
                                    
                                    %% 3.2 取出公共神经元对应的柱号，取出对应的输出
                                    Tout = [];
                                    [comColumns,~] = ind2sub(size(neuronConnectNum),commonNeurons);
                                    comIndex = zeros(size(comColumns));
                                    for comi = 1:length(comColumns)
                                        comIndex(comi) = find(activecolumns == comColumns(comi));
                                    end
                                    Tout= currentTout(comIndex);
                                    
                                    %% 3.3 计算激活神经元输出currentTout的在下个周期的PSP，膜电压，输出
                                    allInputs = Tout' + delay;
                                    simtime = min(allInputs)-T_step:T_step:max(allInputs)+endTime; %下个周期的运行时间
                                    AllV = [];
                                    for t = simtime
                                        %% input
                                        temp = t - allInputs;
                                        temp(temp<=0) = inf;
                                        PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                        V = weight'*PSP;
                                        AllV(end+1) = V;
                                    end
                                    maxV = max(AllV);
                                    [~,maxVT] = find(AllV == maxV);
                                    toutSeg = simtime(maxVT);
                                    
                                    %% 3.4 若有输出证明segments激活了，保存相关信息
                                    if (length(toutSeg) == 1)                          %若被激活
                                        predictState(neuronSegIndex) = 1; %该神经元当前时刻变为预测状态
                                        hasContextNeurons(commonNeurons) = 1;
                                        wordName = segmentW(comIdx,3);               %该segment的标签
                                        if any( wordName ~= wordName(1))
                                            wordName = mode(wordName);
                                        else
                                            wordName =  wordName(1);
                                        end
                                        index = find(predictWordList == wordName);     %查看这个单词是否被预测过
                                        if (isempty(index)) %第一次出现这个预测单词
                                            predictWordList(end+1) = wordName;        %将该单词加入列表
                                            predictInformation{length(predictWordList),1} = toutSeg; %该segment的输出保存，第1列装输出
                                            predictInformation{length(predictWordList),2} = neuronSegIndex; %第2列装对应的神经元
                                        else                %非第一次出现预测的单词
                                            pridictIdx =  predictInformation{index,2};
                                            if isempty(find(pridictIdx == neuronSegIndex))
                                                preTout = predictInformation{index,1};        %找到之前存储的位置
                                                preTout(end+1,1:length(toutSeg)) = toutSeg;%输入拼接上去
                                                preTout(preTout == 0) = simtime(end);      %多出的自动补最大时间
                                                predictInformation{index,1} = preTout;        %第1列装输出
                                                predictInformation{index,2} (end+1,:)= neuronSegIndex; %第2列装对应的神经元
                                            end
                                        end
                                    else
                                        fprintf('多个输出');
                                        fprintf(fileID,'多个输出');
                                    end
                                end
                            end
                        end
                        
                        %% 存储第一个单词的信息
                        if newWordFlag == 1
                            FirstWordInfo{sample,1} = predictState;
                            FirstWordInfo{sample,2} = predictWordList;
                            FirstWordInfo{sample,3} = predictInformation;
                        end
                    end
                    %             if size(FirstWordInfo,1)~=1
                    %                 size(FirstWordInfo,1)
                    %             end
                    preState(:,:,word) = predictState;
                    
                else
                    %% 开始预测
                    fprintf('Test %d sentence: ',seq);
                    fprintf(fileID,'Test %d sentence: ',seq);
                    fprintf('%s ',inputLablesCell{testSeq(seq,1:word-1)});
                    fprintf(fileID,'%s ',inputLablesCell{testSeq(seq,1:word-1)});
                    fprintf('\n');
                    fprintf(fileID,'\n');
                    
                    %%     判断一下是否为叶子结点
                    hasContextNeurons = find(hasContextNeurons~=0);               %有预测的激活神经元
                    
                    if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold)&&length(actIndex)~=activeNum*neuronNum%有预测的激活神经元《〈激活神经元，且训练不只一个单词（第一个单词全激活）
                        predicSegLen = size(outputInfo,1);
                        outputInfo{predicSegLen+1,1} = trainInfo{1};%第1列存该句子所有单词标签
                        outputInfo{predicSegLen+1,2} = trainInfo{2};%第2列存该句子所有单词激活的神经元
                        outputInfo{predicSegLen+1,3} = trainInfo{3};
                    end
                    
                    %% 开始建树预测
                    haveReadList = zeros(size(predictWordList));   %用于存储已经用过的单词
                    while (~isempty(predictWordList))              %已经遍历完了，栈内没有任何元素了
                        %% 弹出最顶上的栈元素
                        listLen = find(haveReadList == 0);         %所有未用过的单词的index
                        
                        %% 栈顶部不存在
                        if(isempty(listLen))
                            predictWordList(listLen) = [];
                            predictInformation(listLen,:) = [];
                            haveReadList(listLen) = [];
                            break;
                        end
                        listLen = listLen(end);                    %最上面那个
                        
                        currentWord = predictWordList(listLen);    %它所在的单词标签
                        %                 inputLablesCell{currentWord}
                        currentSegmentsCell = predictInformation(listLen,1:2);%它所在的单词的预测神经元信息
                        haveReadList(listLen) = 1;                            %该单词已经用过了
                        %% 赋值
                        currentTout = currentSegmentsCell{1,1};    %当前的预测单词所在神经元的实际输出
                        %% 没有输入预测状态变为活跃状态
                        actIndex = currentSegmentsCell{1,2};       %激活神经元所在的index
                        if length(actIndex) < segmentThreshold
                            %% 去掉这个单词，是噪声，会导致栈顶不存在
                            predictWordList(listLen) = [];
                            predictInformation(listLen,:) = [];
                            haveReadList(listLen) = [];
                            continue;
                        end
                        tempWordList = []; %% 当前单词的预测单词
                        tempInformation = {}; %% 当前单词的预测的segments
                        hasContextNeurons = zeros(columnNum,neuronNum);
                        %% step1:遍历所有segment
                        for num = 1:length(neuronHasSegmentsIndexs)
                            neuronSegIndex = neuronHasSegmentsIndexs(num);
                            segments = segmentsList{neuronSegIndex};%获取相应位置的segments
                            %% step2 判断该segments和激活神经元有相交的神经元没有，小于阈值，则不用计算膜电压了
                            for j = 1:segmentsNum(neuronSegIndex) %该神经元segments的个数，因为segments中第二行是标签，所以我们这里用的延迟
                                % 当前segment的连接权重
                                segmentW = segments{1,j};
                                %segmentlabel = segments{2,j};
                                % 权重对应的神经元
                                segmentSynIdx = segmentW(:,1);
                                %% 比较segment有权重的神经元和激活神经元的交集
                                
                                %% 矩阵形式的公用神经元
                                diffNeurons = actIndex-segmentSynIdx';
                                [~,comIdx] = find(diffNeurons==0);
                                
                                
                                %判断该segments的交集是否大于阈值，若小于阈值，则不计算膜电压
                                if (length(comIdx) >= segmentThreshold)
                                    commonNeurons = segmentSynIdx(comIdx);
                                    %% step3 计算segment在下个周期的膜电压，膜电压超过阈值的点火
                                    hasContextNeurons(commonNeurons) = 1;
                                    %% 3.1 取出公共神经元的延迟
                                    delay = segmentW(comIdx,4);              %该神经元对应的segment的延迟
                                    weight = segmentW(comIdx,2);             %segment连接的权重
                                    
                                    %% 3.2 取出公共神经元对应的输出
                                    %要被删除的index
                                    delColumn = setdiff(actIndex,commonNeurons);
                                    %找出要删除柱的下标
                                    delIndex = ismember(actIndex,delColumn);
                                    %删除其他柱的输出，剩下的就是公共神经元的输出
                                    Tout = currentTout;          %输出
                                    Tout(delIndex==1) = [];
                                    
                                    %% 3.3 计算激活神经元输出currentTout的在下个周期的PSP，膜电压
                                    %                             timePeriod = ceil((Tout+delay)/Tmax); %下个周期的时间
                                    %                             if max(timePeriod)-min(timePeriod)==0
                                    %                                 simtime = timePeriod(1)*Tmax + testPeriodicTime;
                                    %                             else
                                    %                                 minT = min(timePeriod);
                                    %                                 maxT = max(timePeriod);
                                    %                                 simtime = minT*Tmax:T_step:(maxT+1.5)*Tmax;
                                    %                             end
                                    
                                    allInputs = Tout + delay;
                                    simtime = min(allInputs)-T_step:T_step:max(allInputs)+endTime;
                                    
                                    AllV = [];
                                    for t = simtime
                                        %% input
                                        temp = t - Tout - delay;
                                        temp(temp<=0) = inf;
                                        PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                        V = weight'*PSP;
                                        AllV(end+1) = V;
                                    end
                                    maxV = max(AllV);
                                    [~,maxVT] = find(AllV == maxV);
                                    toutSeg = simtime(maxVT);
                                    
                                    if (length(toutSeg) == 1)           %若被激活
                                        wordName = segmentW(comIdx,3);               %该segment的标签
                                        if any( wordName ~= wordName(1))
                                            wordName = mode(wordName);
                                        else
                                            wordName =  wordName(1);
                                        end
                                        index = find(tempWordList == wordName);
                                        if (isempty(index)) %第一次出现这类输出
                                            tempWordList(end+1) = wordName;
                                            tempInformation{length(tempWordList),1} = toutSeg; %第1列装输出
                                            tempInformation{length(tempWordList),2} = neuronSegIndex; %第2列装对应的柱神经元
                                        else                %非第一次出现
                                            pridictIdx =  tempInformation{index,2};
                                            if isempty(find(pridictIdx == neuronSegIndex))
                                                preTout = tempInformation{index,1};        %找到之前存储的位置
                                                preTout(end+1,1:length(toutSeg)) = toutSeg;%输入拼接上去
                                                preTout(preTout == 0) = simtime(end);      %多出的自动补最大时间
                                                tempInformation{index,1} = preTout;        %第1列装输出
                                                tempInformation{index,2} (end+1,:)= neuronSegIndex; %第2列装对应的神经元
                                            end
                                        end
                                    else
                                        fprintf('膜电压不够没有输出\n');
                                        fprintf(fileID,'膜电压不够没有输出\n');
                                    end
                                end
                            end
                        end
                        %% step3 判断是否为叶子结点叶子结点1：判断当前是否有激活的神经元没有预测的，表明句子结束，加入列表
                        leaf = 0;
                        hasContextNeurons = find(hasContextNeurons~=0);                %有预测的激活神经元
                        %预测的激活神经元<当前的激活神经元，且不是全激活的单词导致的
                        if (length(actIndex) - length(hasContextNeurons) >= segmentThreshold)
                            leaf = 1;
                        end
                        %% 判断当前有没有新的预测，若有，那么加入栈，若没有，那么是叶子结点
                        nextLen = length(tempWordList);
                        if nextLen > 0 %%非叶子结点
                            predictWordList(end+1:end+nextLen) = tempWordList;
                            predictInformation(end+1:end+nextLen,:) = tempInformation;
                            haveReadList(end+1:end+nextLen) = 0;
                        else
                            leaf = 1; %% 无预测,叶子结点2，整体都没有输出
                        end
                        if leaf == 1
                            %% 结束 保存所有词的标签
                            wordList = find(haveReadList == 1);
                            
                            predicSegLen = size(outputInfo,1);
                            if (predicSegLen > maxPredicSen)
                                break;
                            end
                            outputInfo{predicSegLen+1,1} = [trainInfo{1},predictWordList(wordList)];%第1列存该句子所有单词标签
                            outputInfo{predicSegLen+1,2} = [trainInfo{2};predictInformation(wordList,2)];%第2列存该句子所有单词激活的神经元
                            outputInfo{predicSegLen+1,3} = [trainInfo{3};predictInformation(wordList,1)];%第3列激活的神经元的输出
                            %% 删除该句子相关信息
                            start = find(haveReadList == 0); %%从还没开始遍历的结点
                            
                            if isempty(start)
                                predictWordList = [];
                                predictInformation = {};
                                haveReadList = [];
                            else
                                start = start(end);
                                predictWordList(start+1:end) = [];
                                predictInformation(start+1:end,:) = [];
                                haveReadList(start+1:end) = [];
                            end
                        end
                        
                    end
                    
                    
                    %% 输出整体的输出
                    %% 方案1：根据输入直接抑制那些不包含输入信息的目标神经元
                    predicStnIdx = [];
                    inputLen = length(find(testSeq(seq,:)~=-1));
                    for i = 1:size(outputInfo,1)
                        fprintf('part%4d:',i);
                        fprintf(fileID,'part%4d:',i);
                        info = outputInfo(i,:);
                        predictSentence = outputInfo{i,1};
                        fprintf(' %s',inputLablesCell{predictSentence});
                        fprintf(fileID,' %s',inputLablesCell{predictSentence});
                        
                        
                        haveGlobalFlag = 0;                 %局部是否激活目标神经元
                        targetV = zeros(seqenceNum,1);
                        minTout = Inf*ones(seqenceNum,1);
                        minSimility = length(unique(predictSentence))/maxSenLen;
                        for goalNeuron = 1:seqenceNum                     %遍历所有目标神经元
                            targetScentence = targetsCell{goalNeuron,3};  %目标神经元的单词序列
                            simility = length(intersect(predictSentence,targetScentence))/length(targetScentence);%比较预测的和目标句子相似度
                            
                            if simility >=minSimility
                                %存在相似度，就去判断一下是否激活
                                
                                targetWeights = targetsCell{goalNeuron,1};%目标神经元和激活神经元相连的权重
                                targetDelays = targetsCell{goalNeuron,2}; %目标神经元和激活神经元相连的权重
                                
                                currentActiveNeurons = outputInfo{i,2};%激活神经元
                                currentTout = outputInfo{i,3};         %激活神经元的输出
                                
                                weights = [];  %存放激活神经元到输出的权重
                                delays = [];   %存放激活神经元到输出的延迟
                                Tout = [];     %存放激活神经元到输出的输出
                                
                                %% 目标神经元和激活神经元相连的连接，延迟，以及输出，预测的和输入的同样处理
                                for k = 1:length(currentActiveNeurons)
                                    idx = currentActiveNeurons{k};
                                    weights = [weights;targetWeights(idx)];
                                    delays  = [delays;targetDelays(idx)];
                                    Tout = [Tout;currentTout{k}];
                                end
                                
                                AllV = [];
                                %% 选择模拟时间
                                allInputs = Tout + delays;
                                allInputs(allInputs ==Inf) = [];
                                sortInputs = sort(round(allInputs));
                                simtime = min(sortInputs):T_step:max(sortInputs)+endTime;
                                %toutSeg = Inf;
                                if isempty(simtime)
                                    AllV = 0;
                                else
                                    for t = simtime
                                        %% input
                                        temp = t - Tout - delays;
                                        temp(temp<=0) = inf;
                                        PSP = Vnorm* sum( (exp(-temp/Tau_m)-exp(-temp/Tau_s)), 2);
                                        %                             ref = Theta*sum(-exp((toutSeg-t)/Tau_a),2);
                                        V = weights'*PSP;
                                        AllV(end+1) = V;
                                        %                             if (V >= targetThreshold)
                                        %                                 haveGlobalFlag = haveGlobalFlag+1;
                                        %                                 toutSeg = t;
                                        %                                 break;
                                        %                             end
                                    end
                                end
                                targetV(goalNeuron) = max(AllV);
                                %                         minTout(goalNeuron) = toutSeg;
                            end
                        end
                        maxV = max(targetV);
                        if maxV < predictNoneThreshold
                            fprintf('\n \n');
                            fprintf(fileID,'\n \n');
                            continue;
                        elseif maxV >= predictHasThreshold
                            maxV = maxV - diffThresholdMany; %多个输出
                        else
                            maxV = maxV - diffThresholdNone;
                        end
                        idx = find(targetV >= maxV);
                        targetScentence = targetsCell(idx,3);
                        predicStnIdx(end+1:end+length(idx)) = idx;
                        fprintf('\n');
                        fprintf(fileID,'\n');
                        for sentenceNum = 1:length(idx)
                            fprintf('out%5d:',idx(sentenceNum));
                            fprintf(fileID,'out%5d:',idx(sentenceNum));
                            fprintf(' %s',inputLablesCell{targetScentence{sentenceNum}});
                            fprintf(fileID,' %s',inputLablesCell{targetScentence{sentenceNum}});
                            fprintf('\n');
                            fprintf(fileID,'\n');
                        end
                        fprintf('\n');
                        fprintf(fileID,'\n');
                    end
                    
                    %% 和真实的相比的正确率
                    predicStnIdx = unique(sort(predicStnIdx)); %预测的句子
                    actualStnIdx = [];
                    subsequence = testSeq(seq,:);       %当前子序列
                    subsequence(subsequence == -1) = []; %去掉-1的那部分
                    subLen = length(subsequence);       %表示子序列的长度
                    for goalNeuron = 1:seqenceNum                     %遍历所有目标神经元
                        targetScentence = targetsCell{goalNeuron,3};  %目标神经元的单词序列
                        targetLen = length(targetScentence);          %表示目标序列的长度
                        firstIdx = find(targetScentence == subsequence(1));%第一个单词在目标序列中的位置
                        
                        if (subLen>targetLen || isempty(firstIdx))    %不是目标的子序列
                            continue;
                        end
                        
                        for i = 1:length(firstIdx) %遍历第一个单词在目标序列中的位置
                            targetIdx = firstIdx(i)+1; %获取第2个单词的位置
                            rightFlag = 1; %假设该句子是
                            for subIdx = 2:subLen  %从第二个单词开始看是否匹配子序列
                                if targetIdx<=targetLen && subsequence(subIdx) == targetScentence(targetIdx)  %若目标单词和子序列一致
                                    targetIdx = targetIdx+1;
                                else %存在一个不一致，那么就不是
                                    rightFlag = 0;
                                    break;%不是子序列
                                end
                            end
                            if rightFlag == 1
                                actualStnIdx(end+1) = goalNeuron;
                                break;
                            end
                        end
                    end
                    %% 计算正确率
                    
                    accuracyIdx = find(actualStnIdx-predicStnIdx'==0);
                    accuracyLen = length(accuracyIdx);
                    unionLen = length(actualStnIdx) + length(predicStnIdx) - accuracyLen;
                    seqAcc(seq,testN) = accuracyLen/unionLen;
                    
                    fprintf('predict:%s\n actual:%s\n',num2str(predicStnIdx),num2str(actualStnIdx));
                    fprintf(fileID,'predict:%s\n actual:%s\n',num2str(predicStnIdx),num2str(actualStnIdx));
                    fprintf('current accuracy:%f, mean accuracy:%f\n',seqAcc(seq,testN),mean(seqAcc(1:seq,testN)));
                    fprintf(fileID,'current accuracy:%f, mean accuracy:%f\n',seqAcc(seq,testN),mean(seqAcc(1:seq,testN)));
                    
                    if seqAcc(seq,testN)~=1
                        if accuracyLen < length(actualStnIdx)
                            diffData = setdiff(actualStnIdx,predicStnIdx);
                            fprintf('absence:');
                            fprintf(fileID,'absence:');
                            for i = 1:length(diffData)
                                fprintf('%d ',diffData(i));
                                fprintf(fileID,'%d ',diffData(i));
                            end
                        end
                        
                        if length(predicStnIdx) > length(actualStnIdx)
                            fprintf('\nsurplus:');
                            fprintf(fileID,'\nsurplus:');
                            diffData = setdiff(predicStnIdx,actualStnIdx);
                            diffData = sort(diffData);
                            for i = 1:length(diffData)
                                fprintf('%d ',diffData(i));
                                fprintf(fileID,'%d ',diffData(i));
                            end
                        end
                        fprintf('\n');
                        fprintf(fileID,'\n');
                    end
                    break;
                end
                
            end
            fprintf('\n');
            fprintf(fileID,'\n');
        end
    end
    fprintf('the mean accuracy of %d trails:%f\n',trail,mean(mean(seqAcc(1:testSeqNum,:))));
    fprintf(fileID,'the mean accuracy of %d trails:%f\n',trail,mean(mean(seqAcc(1:testSeqNum,:))));
    str(end-5:end)=[];
    save([str,'.mat'],'seqAcc');
end


