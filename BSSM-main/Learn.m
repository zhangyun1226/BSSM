clear;
close all;
%% 时间池代码
initParameter;
load data1000.mat;
load spation1000.mat
neuronNum = 32;
sysinput = 100;
seqenceLen = size(sequence,1);                %训练的序列
wordLen = size(sequence,2);                   %每个序列的最大长度
seqenceLen = 1000;                                       %训练的句子数量
%% 先将突触前的电压算出，方便计算
inputVCell = {};
for wordNum = 1: max(sequence(:))
    tempPSP = squeeze(allPSPTrain(wordNum,:,:));
    inputVCell{wordNum} = inputWeight'*tempPSP;
end
%% segment 相关参数
segmentConnect = 7;                       %每个segment和激活神经元相连的个数
maxSegmentConnect = 4*segmentConnect;     %每个segments最多记忆4个上文，且不要共用
targetWordNum = 4;                        %目标单词4个可以使目标神经元膜电压到达1
wi = 1/(activeNum*targetWordNum);         %每个激活神经元到目标神经元的权重
firstWi = 1/(activeNum*neuronNum*targetWordNum);
segmentThreshold = segmentConnect;
disWeight = 1/(segmentThreshold);
columnInhW = 0.25;

for num = 1:columnNum*neuronNum
    segmentsList{num} = {};%存储权重
end
for num = 1:columnNum*neuronNum
    connectedSegmentList{num} = {};%存储权重
end
segmentsNum = zeros(columnNum,neuronNum);              %每个神经元有几个segment
neuronConnectNum = zeros(columnNum,neuronNum);         %每个神经元向外连接的次数
DelayTime = Tau_m*Tau_s*log(Tau_m/Tau_s)/(Tau_m-Tau_s);%延迟到达最大PSP的时间
%% 训练
targetWeights = zeros(columnNum,neuronNum,seqenceLen);  %到目标神经元的权重
targetDelays = Inf*ones(columnNum,neuronNum,seqenceLen);%到目标神经元的延迟
targetNeurons =  zeros(columnNum,neuronNum,seqenceLen);
% secondShare = [6,8,11,12,17,18,20,21];
for seq = 1:1:seqenceLen                               %遍历训练序列
    tempTargetWeights = zeros(columnNum,neuronNum);  %到目标神经元的权重
    tempTargetDelays = Inf*ones(columnNum,neuronNum);%到目标神经元的延迟
    tempTargetNeurons = zeros(columnNum,neuronNum);
    sentenceLen = length(sequence(seq,:)~=0);
    for wordNum = 1:wordLen                            %遍历每个单词
        %% 记录上一个单词的位置
        if wordNum~=1
            lastWord = word;
            allInputVlast = inputVCell{lastWord};
        else
            lastWord = 0;
            allInputVlast = zeros(columnNum,nt);
        end
        %% 当前单词的操作
        word = sequence(seq,wordNum);               %当前样本
        %% 初始化膜电压和神经元的输出
        toutCA3Neuron = -Inf*ones(columnNum,neuronNum);
        fireColumns = zeros(columnNum,1);
        fireNeurons = zeros(columnNum,neuronNum);
        intColumnTime = Inf*ones(columnNum,neuronNum);
        zerosV = zeros(columnNum,neuronNum);
        
        if word == 0 %%序列结束
            targetWeights(:,:,seq) = tempTargetWeights;
            targetDelays(:,:,seq) = tempTargetDelays;
            targetNeurons(:,:,seq) = tempTargetNeurons;
            break;
        else
            fprintf("seq %d word %d:%s\n",seq,wordNum,inputLablesCell{word});
            %% 计算当前的每个神经元的输出,只保留了最近一次的输出
            %% 输入分为前馈输入 抑制输入 振荡输入 侧向输入 不应期
            startTime = (wordNum-1)*nt;      % 起始时间
            winnerSegIdx = zerosV;           % 用于存储哪个树突让神经元点火
            winnerSegV = zerosV;             % winner树突的膜电压
            allInputV = inputVCell{word};    % 当前单词的前馈电压
            allV = [];                       % 存储膜电压，分析用
            %% 确定哪些远端树突参与远端计算
            disSegNeu = [];                  % 计算哪些神经元有segment，
            disSegNum = [];                  % 神经元的第几个segments接收上个周期传递过来的信息
            if wordNum == 2
                %% 第一个单词激活柱上的所有神经元【column,neuron]
                tempNeu = [];
                for tempi = 1:length(targetColumnsLast)
                    tempNeu = [tempNeu,targetColumnsLast(tempi)+(0:neuronNum-1)*columnNum];
                end
                %% 将这些神经元连接的下文segments加入计算列表
                connectedNeurons = [];           %哪些神经元与上个周期的神经元有连接
                for tempi = 1:length(tempNeu)
                    tempNeuron = tempNeu(tempi); %获取上个周期的神经元
                    tempSegments = connectedSegmentList{tempNeuron};%与它连接的segments加入connectedNeurons
                    if ~isempty(tempSegments)
                        connectedNeurons = [connectedNeurons;tempSegments];
                    end
                end
                %% 去重（某个segment与多个神经元相连，所以会出现多次）
                if ~isempty(connectedNeurons)
                    disSegNeu = connectedNeurons(:,1); %有远端树突的神经元
                    disSegNum = connectedNeurons(:,2); %远端树突的编号
                    [~,idx]= unique(disSegNeu*1000+disSegNum); %编码后去重
                    disSegNeu = disSegNeu(idx);
                    disSegNum = disSegNum(idx);
                end
            elseif wordNum>2
                connectedNeurons = [];
                for tempi = 1:length(lastLearnNeurons)
                    tempNeuron = lastLearnNeurons(tempi);
                    tempSegments = connectedSegmentList{tempNeuron};
                    if ~isempty(tempSegments)
                        connectedNeurons = [connectedNeurons;tempSegments];
                    end
                end
                %% 若之前的学习神经元有连接的下文
                if ~isempty(connectedNeurons)
                    disSegNeu = connectedNeurons(:,1); %有远端树突的神经元
                    disSegNum = connectedNeurons(:,2);
                    [~,idx]= unique(disSegNeu*1000+disSegNum);
                    disSegNeu = disSegNeu(idx);
                    disSegNum = disSegNum(idx);
                end
            end
            %% 开始计算膜电压
            intColumnTime = Inf*ones(columnNum,1);%抑制神经元的输出
            toutCA3Neuron = -Inf*ones(columnNum,neuronNum);%神经元在该周期最近的一次点火
            fireColumns = zeros(columnNum,1);              %哪些柱点火
            fireNeurons = zeros(columnNum,neuronNum);      %哪些神经元点火
            for t = startTime+(through:nt)*dt  %算该神经元的点火时间，遍历through到Tmax
                tIdx = t/dt;                   %当前时刻
                %% step 1 前馈输入的产生的前馈电压
                currentT = tIdx-startTime;
                inputV = allInputV(:,currentT);
                %% step 2 神经元的抑制 = 柱的抑制
                tempInh = t - intColumnTime;
                tempInh(tempInh<=0) = inf;
                CA3I = ca3Ivalue*sum(exp(-tempInh/Tau_s),2);
                %% step 3 神经元的振荡 = 柱的振荡，1+2+3为柱上的所有输入
                columnV = inputV + CA3I + allOsc(:,currentT);
                shareV = repmat(columnV,1,neuronNum);%扩充维度，计算每个神经柱上所有神经元的膜电压
                %% step 4 神经元自己独特的性质，远端树突+不应期+柱内抑制
                CA3NeuV = zerosV;       %初始化每个神经元的独特电压
                tempSegmIdx = zerosV; %每个神经元获胜segment的位置
                %4.1 神经元的不应期
                ref = -exp((toutCA3Neuron-t)/Tau_a);
                %4.2 神经元的远端树突产生的电压,多个远端树突只有电压最大的获胜
                for i = 1:length(disSegNeu)
                    neu = disSegNeu(i);               %第i个神经元neu
                    segi = disSegNum(i);              %neu神经元的segment位置
                    segment = segmentsList{neu}{segi};%对应的segment
                    %% 计算当前segment的电压
                    prei = segment(:,1);              %突触前神经元
                    delayi = segment(:,4);            %延迟
                    weighti = segment(:,2);           %权重
                    pretout = toutCA3NeuronLast(prei);%输出
                    %与突触前神经元的电压
                    temp = t - pretout - delayi;
                    temp(temp<=0) = Inf;
                    tempPSP = exp(-temp/Tau_s);
                    segV = weighti'*tempPSP;
                    %如果这个膜电压大于之前的树突，则保留下来替换
                    if segV > CA3NeuV(neu)
                        CA3NeuV(neu) = segV;             %该神经元的远端膜电压
                        tempSegmIdx(neu) = segi;         %对应的segment
                    end
                end
                tempSegmementV = CA3NeuV;
                % 4.3 柱内神经元间的抑制加入CA3NeuV
                allFireColumns = find(fireColumns >0);   %点火的柱
                fileColumnsLen = length(allFireColumns); %点火柱的数目
                if fileColumnsLen>0
                    if fileColumnsLen*neuronNum ~= length(find(fireNeurons)>0)%若不是全点火
                        innerInhV = -columnInhW*ones(fileColumnsLen,neuronNum);%点火柱内所有神经元都被抑制
                        CA3NeuV(allFireColumns,:) = CA3NeuV(allFireColumns,:) + innerInhV;
                        CA3NeuV(fireNeurons~=0) = CA3NeuV(fireNeurons~=0)+columnInhW;%点火的神经元不被抑制
                    end
                end
                %% 计算神经元的膜电压
                CA3EV = shareV + CA3NeuV + ref;
                allV(end+1,:) = CA3EV(:);
                %                 AllColumnV(tIdx,:) = columnV(:);
                %                 CA3IV(end+1,:) = CA3I(:);
                %                 refV(end+1,:) = ref(:);
                %% 存在超过阈值的神经元
                neuron = find(CA3EV >=Theta);      %膜电压是否大于阈值点火
                if(~isempty(neuron))               %若存在神经元点火
                    %寻找神经元对应的柱
                    [column,~] = ind2sub(size(CA3EV),neuron);
                    %1.柱的抑制时间增加一维
                    currentOut = t*ones(columnNum,1);
                    currentOut(column) = Inf;
                    if all(intColumnTime == Inf)
                        intColumnTime = currentOut;
                    else
                        intColumnTime(:,end+1) = currentOut;
                    end
                    %2.神经元的输出变化，只保存最近一次的
                    toutCA3Neuron(neuron) = t;
                    %3.点火的柱和神经元发生变化，并记住它们的获胜segment
                    fireColumns(column) = 1; %柱点火的标志
                    fireNeurons(neuron) = 1; %神经元点火的标志
                    winnerSegIdx(neuron) = tempSegmIdx(neuron); %获胜segment的位置
                    winnerSegV(neuron) = tempSegmementV(neuron);%每个segment的最大远端电压
                end
            end
            %% 实际点火的柱，目标柱：用目标柱的神经元建立连接
            targetColumn = outputColumn(word,:);   %目标柱
            %% 在波峰建立远端树突-(第一个单词除外)
            if wordNum ==1
                lastLearnNeurons = [];             %第一个单词，没有学习神经元，不用建立树突
                targetColumnsLast = targetColumn;    %保存这个周期柱的输出
                
                lastFireNeurons = [];              %保存每个目标神经元的输出
                for tempi = 1:length(targetColumnsLast)
                    lastFireNeurons = [lastFireNeurons,targetColumnsLast(tempi)+(0:neuronNum-1)*columnNum];
                end
                
                toutCA3NeuronLast = toutCA3Neuron; %保存这个周期神经元的输出
                continue;
            else %其他单词
                %% 初始化学习神经元
                learnNeurons = zeros(1,activeNum);
                %% 获取每个柱的点火神经元情况，按点火的数量降序排列柱
                winnerNum = winnerSegIdx(targetColumn,:); %目标柱上的获胜神经元对应的segment
                winnerNum(winnerNum>1)=1;                 %只要不是0就是存在获胜segment
                winnerNum = sum(winnerNum,2);             %每个柱上的有获胜segment的数量
                winnerNum(winnerNum == 0) = neuronNum;    %一个都没有，算做全激活
                [winnerNum,winIdx] = sort(winnerNum,'descend');
                targetColumn = targetColumn(winIdx);
                %% 若存在柱全激活，且上个周期的柱没有选择学习神经元
                if isempty(lastLearnNeurons) && winnerNum(1) == neuronNum
                    %% 选择向外连接数最少的作为学习神经元
                    preNeurons = zeros(1,activeNum);                        %为上一个周期选择神经元
                    for num = 1:activeNum
                        tempColumn = targetColumnsLast(num);                %遍历上个周期的目标柱
                        neronsCon = neuronConnectNum(tempColumn,:);         %柱的神经元向外的连接数目
                        minConNeuron = find(neronsCon == min(neronsCon));   %选择最少连接的数的神经元
                        currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %可能有多个，随机选择一个神经元
                        preNeurons(num) = currentNeuron;
                    end
                    %% 将神经元转换为对应的index
                    lastLearnNeurons = sub2ind(size(neuronConnectNum),targetColumnsLast,preNeurons);
                end
                
                %% 从点火数最多的柱开始遍历建立树突
                noiseFlag = 0;
                for i = 1:activeNum
                    %% step 1 选择学习神经元，判断当前点火柱的情况，有获胜柱和部分点是旧知识
                    currentColumn = targetColumn(i);   %获取当前激活柱
                    chooseFlag = 0;
                    %% 没有获胜的神经元，无旧知识，选择树突最少的神经元为学习神经元
                    if winnerNum(i)==neuronNum || noiseFlag == 1
                        neurons = find(segmentsNum(currentColumn,:) == min(segmentsNum(currentColumn,:)));
                        neuronsIdx = find(toutCA3Neuron(neurons) ==min(toutCA3Neuron(neurons)));
                        chooseNeuron2 = neurons(neuronsIdx(unidrnd(length(neuronsIdx))));%有多个就随机选择一个
                        chooseNeuron = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron2);
                        chooseFlag = 1;
                    else
                        %% 存在获胜树突的神经元，有旧知识或者噪声
                        chooseNeuron2 = find(winnerSegIdx(currentColumn,:)>0); %找到哪些神经元有获胜树突
                        if length(chooseNeuron2) > 1 %若多个，选择树突电压最大的那个为学习神经元
                            chooseNeuronV = winnerSegV(currentColumn,chooseNeuron2);
                            [~,neuronIdx] = max(chooseNeuronV);
                            chooseNeuron2 = chooseNeuron2(neuronIdx);
                        end
                        chooseNeuron = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron2);
                        winnerIdx = winnerSegIdx(chooseNeuron); %学习神经元的获胜树突位置
                        winnerSegment = segmentsList{chooseNeuron}{winnerIdx};%学习神经元的获胜树突位置
                        chooseFlag = 2;
                    end
                     %% step 2 为学习神经元建立树突/连接
                        ...1.判断winner segments上和学习神经元的相似度，在此基础上建立连接
                        ...2.不存在winner-segment，新建一个树突和神经元建立连接
                    if chooseFlag == 2
                        prei = winnerSegment(:,1);
                        if isempty(lastLearnNeurons) %如果上个周期是第一个单词，计算上周期的期望柱的连接数
                            [connectColumns,~] = ind2sub(size(neuronConnectNum),prei);%突触前神经元所在柱
                            diffColumns = connectColumns - targetColumnsLast;
                            [~,comColumnIdx] = find(diffColumns==0);%比较和上个周期的柱的连接数
                            comColumn = unique(targetColumnsLast(comColumnIdx));
                            %% 噪声导致的点火
                            if length(comColumn)<segmentConnect %连接数小于阈值，噪声导致的点火，需要重新选择上个周期的学习神经元
                                %% 重新选择上个周期的学习神经元
                                preNeurons = zeros(1,activeNum);
                                for num = 1:activeNum                                            %遍历激活柱
                                    currentColumn = targetColumnsLast(num);                         %激活柱i
                                    neronsCon = neuronConnectNum(currentColumn,:);               %柱i的神经元向外的连接数目
                                    minConNeuron = find(neronsCon == min(neronsCon));            %选择最少连接的数的神经元
                                    currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %可能有多个，随机选择一个神经元
                                    preNeurons(num) = currentNeuron;
                                end
                                %% 将神经元转换为对应的index
                                lastLearnNeurons = sub2ind(size(neuronConnectNum),targetColumnsLast,preNeurons);
                                %% 选择其他神经元新建连接
                                chooseFlag = 1;
                                noiseFlag = 1;
                                neurons = find(segmentsNum(currentColumn,:) == min(segmentsNum(currentColumn,:)));
                                neuronsIdx = find(toutCA3Neuron(neurons) ==min(toutCA3Neuron(neurons)));
                                chooseNeuron2 = neurons(neuronsIdx(unidrnd(length(neuronsIdx))));%有多个就随机选择一个
                                chooseNeuron = sub2ind(size(neuronConnectNum),currentColumn,chooseNeuron2);
                                %                                 secondActiveNum(seq) = 32;
                            else
                                %% 连接数大于等于阈值，是上文导致的点火，树突无需改变～
                                learnNeurons(i) = chooseNeuron; %将第i柱的学习神经元赋值
                                continue;  %% 第二个单词完全一致
                            end
                        else
                            %% 比较突触前神经元和学习神经元的相似度
                            diffNeurons = prei - lastLearnNeurons;
                            [~,comIdx] = find(diffNeurons==0);
                            % 3.连接数量大于等于阈值，不需要改树突
                            if length(comIdx)>=segmentConnect
                                learnNeurons(i) = chooseNeuron; %将第i柱的学习神经元赋值
                                continue; %% 第2+n个单词完全一致
                            elseif length(comIdx)<4&&length(comIdx)>0 %连接数量小于一半，新增7-x个连接，
                                segmentLen = size(winnerSegment,1);       %当前segment包含的神经元个数
                                addLen = segmentConnect - length(comIdx); %需要新增加的树突数目
                                if addLen+segmentLen > maxSegmentConnect  %若加上树突后超过了树突的容量
                                    chooseFlag = 1; %新建树突
                                else %若加上树突后没超过树突的容量，则在该树突上修改
                                    %% 已知共用的树突连接comIdx，要往里加东西
                                    preNeuronsIdx = setdiff(1:activeNum, comIdx); %未被连接的上周期学习神经元
                                    preNeuronsIdx = preNeuronsIdx(randperm(length(preNeuronsIdx),addLen)); %从这些神经元中随机选择addLen个
                                    preNeurons = lastLearnNeurons(preNeuronsIdx);
                                    
                                    postTout = toutCA3Neuron(chooseNeuron);    %当前拥有树突的神经元输出
                                    preTouts = toutCA3NeuronLast(preNeurons);  %突触前神经元的输出
                                    delay = postTout - (preTouts + DelayTime); %计算延迟
                                    %% 将相关信息加入树突中
                                    winnerSegment( segmentLen+1: segmentLen+addLen,1) = preNeurons;%2.3.1segments的上文的神经元的index
                                    winnerSegment( segmentLen+1: segmentLen+addLen,2) = disWeight;  %2.3.2到segments的上文的神经元的权重
                                    winnerSegment( segmentLen+1: segmentLen+addLen,3) = word;     %2.3.3当前单词的信息
                                    winnerSegment( segmentLen+1: segmentLen+addLen,4) = delay;
                                    %% 将修改的segment加入segment列表
                                    segmentsList{chooseNeuron}{winnerIdx} = winnerSegment;          %更新树突列表里的信息
                                    neuronConnectNum(preNeurons) = neuronConnectNum(preNeurons) +1; %突触前向外神经元的连接数增加1
                                    %% 将新加突触前神经元的后文segment加上
                                    for tempi = 1:addLen
                                        temp = connectedSegmentList{preNeurons(tempi)};
                                        if isempty(temp)
                                            temp = [chooseNeuron,winnerIdx];
                                        else
                                            temp(end+1,:) = [chooseNeuron,winnerIdx];
                                        end
                                        connectedSegmentList{preNeurons(tempi)} = temp;
                                    end
                                end
                            else %连接数大于等于4，是噪声导致的点火，新建树突
                                chooseFlag = 1; %新建树突
                            end
                        end
                    end
                    
                    if chooseFlag == 1 %新建树突
                        %% 虽然有上文，但是上文错误，需要重新找上文的学习神经元
                        if isempty(lastLearnNeurons) %若上个周期的神经元是第一个周期，需要选择上个周期的学习神经元
                            preNeurons = zeros(1,activeNum);
                            for num = 1:activeNum                                            %遍历激活柱
                                currentColumn = targetColumnsLast(num);                         %激活柱i
                                neronsCon = neuronConnectNum(currentColumn,:);               %柱i的神经元向外的连接数目
                                minConNeuron = find(neronsCon == min(neronsCon));            %选择最少连接的数的神经元
                                currentNeuron = minConNeuron(unidrnd(length(minConNeuron))); %可能有多个，随机选择一个神经元
                                preNeurons(num) = currentNeuron;
                            end
                            %% 将学习神经元转换为对应的index
                            lastLearnNeurons = sub2ind(size(neuronConnectNum),targetColumnsLast,preNeurons);
                        end
                        % 1. 初始化树突的内容
                        segment = zeros(segmentConnect,4);
                        % 2. 上一个周期的10个学习神经元，随机选择70%做为连接的神经元
                        preNeurons = lastLearnNeurons(randperm(activeNum,segmentConnect));
                        % 3. 将信息填入新建树突
                        segment(:,1) =  preNeurons;              %填入突触前神经元
                        preTouts = toutCA3NeuronLast(preNeurons);%突触前神经元的输出
                        postTout = toutCA3Neuron(chooseNeuron);  %当前树突神经元的输出
                        if postTout == -Inf %若当前神经元没有输出
                            postTout = startTime+through+find(allV(:,chooseNeuron)==max(allV(:,chooseNeuron)));
                        end
                        delay = postTout - (preTouts + DelayTime); %计算延迟
                        segment(:,4) = delay;                      %填入突触前神经元延迟
                        segment(:,2) = disWeight;
                        segment(:,3) = word;
                        %% 将新建的segment加入segment列表
                        num = segmentsNum(chooseNeuron);            %当前学习神经元拥有的segments的数量
                        currentSegCell = segmentsList{chooseNeuron};%当前学习神经元拥有的segments
                        currentSegCell{num+1} = segment;            %将新segment写入
                        segmentsList{chooseNeuron} = currentSegCell;%更新信息
                        neuronConnectNum(preNeurons) = neuronConnectNum(preNeurons) +1; %突触前向外神经元的连接数增加1
                        segmentsNum(chooseNeuron) = num+1;                %该神经元的segment数加1
                        %% 突触前神经元也存
                        for tempi = 1:segmentConnect
                            temp = connectedSegmentList{preNeurons(tempi)};
                            if isempty(temp)
                                temp = [chooseNeuron,num+1];
                            else
                                temp(end+1,:) = [chooseNeuron,num+1];
                            end
                            connectedSegmentList{preNeurons(tempi)} = temp;
                        end
                    end
                    %% 树突建立修改完毕
                    learnNeurons(i) = chooseNeuron; %将第i柱的学习神经元赋值
                end
                %% 每个柱上的神经元其树突建立修改完毕
                
            end
            
            %% 保留这个周期的相关信息
            lastLearnNeurons = learnNeurons;
            targetColumnsLast = targetColumn;
            %！！！只保留学习神经元的输出
            toutCA3NeuronLast = -Inf*ones(columnNum,neuronNum);
            learnNeuronsOut = toutCA3Neuron(learnNeurons);
            if find(learnNeuronsOut==-Inf)
                tempIdx = find(learnNeuronsOut==-Inf);
                tempNeurons = learnNeurons(tempIdx);
                [~,maxT] = max(allV(:,tempNeurons));
                learnNeuronsOut(tempIdx) = maxT+1+(startTime+through);
            end
            toutCA3NeuronLast(learnNeurons) = learnNeuronsOut;
            lastFireNeurons = find(fireNeurons == 1);
            tempTargetWeights(learnNeurons) = wi;            %第i个单词的学习神经元到目标的权重
            tempTargetDelays(learnNeurons) = (sentenceLen+1 - wordNum)*T;%第i个单词的激神经元到目标的延迟；nT,(n-1)T...T
            tempTargetNeurons(learnNeurons) = 1;
        end
        
    end
end
tempralStr = ['temporalout',num2str(neuronNum)];
tempralStr = [tempralStr,'.mat'];
save(tempralStr);
testReplace;