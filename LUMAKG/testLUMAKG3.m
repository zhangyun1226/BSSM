%% testing
%% 将输入词转换为初始序列
for j = 1: length(input)
    new_word = string(input{j});
    for k = 1: nc
        if strcmp(column(k).word, new_word)
            p = k;
            break;
        end
    end
    if ~isempty(p)
        WMS(j) = p;
        if j == 1
            path = cell(M, 1);      % 初始路径
            for i = 1: M
                path{i} = [WMS(j) i];
            end
        else
            np = 0;
            new_path = [];
            for i = 1: length(path)
                WM = path{i};
                suc = column(WM(end, 1)).neuron(WM(end, 2)).successor;
                if ~isempty(suc)
                    for m = 1: M
                        if issubset([WMS(j) m], suc(:, 1:2))
                            np = np + 1;
                            new_path{np} = [WM; WMS(j) m];
                        end
                    end
                end
            end
            if ~isempty(new_path)    % 如果输入一个句子的第一个词和第三个词该怎么办？
                path = new_path;
            else   % 原训练集中不存在当前输入词与之前输入词的序列
                for i = 1: M
                    path{i+1} = [WMS(j) i];
                end
            end
        end
    end
end
%% 恢复序列
while ~isempty(path)
    np = 0;         % 新一轮路径数量
    new_path = [];
    for i = 1: length(path)
        flag = 1;
        WM = path{i};
        nw = size(WM, 1);
        suc = column(WM(end, 1)).neuron(WM(end, 2)).successor;
        if ~isempty(suc)
            for j = 1: size(suc, 1)
                presyn = column(suc(j,1)).neuron(suc(j,2)).precessor;
                ti = [];    w = [];
                for k = 1: size(WM, 1)
                    [flag, loc] = issubset(WM(k,:), presyn(:, 1:2));
                    if  flag
                        ti = [ti; k];
                        w = [w; presyn(loc, 4)];
                    end
                end
                if isempty(ti)
                    continue
                end
                idx = find(ti==nw+1);
                ti(idx)=[];
                w(idx) = [];
                if isempty(w)||nw>=20
                    continue;
                end
                ti = dT(nw) - dT(nw+1-ti);
                Timeline = 0: dT(nw) + t_max;%(epo-1) * dT + 1 + t_max + 1; %  ti';
                temp = Timeline - ti;
                temp(temp <= 0) = Inf;
                psp = Vnorm * ( exp( -temp / tau_m ) - exp( -temp / tau_s ) ); % = exp(- temp ./ tau_m);
                membrane = w' * psp;
                col = find( membrane >= theta );
                if ~isempty(col) %&& Timeline(col) < (epo-1) * dT + 1
                    np = np + 1;
                    new_path{np} = [WM; suc(j,1:2)];
                end
            end
        end
    end
    if isempty(new_path)
        testsequence = {};
        for targetIdx = 1:length(path)
            testseq = [];
            for j = 1: size(path{targetIdx}, 1)
                str = column(path{targetIdx}(j, 1)).word;
                fprintf('%s ', str);
                fprintf(frepId,'%s ',str);
                for tempk = 1:strLen
                    if strcmp(strCell{tempk},str)
                        break;
                    end
                end
                testseq(end+1) = tempk;
            end
            testsequence{targetIdx} = testseq;
            fprintf('\n');
            fprintf(frepId,'\n');
        end
        
    end
    path = new_path;
    if length(path)>20
        testsequence = {};
        for targetIdx = 1:length(path)
            testseq = [];
            for j = 1: size(path{targetIdx}, 1)
                str = column(path{targetIdx}(j, 1)).word;
                fprintf('%s ', str);
                fprintf(frepId,'%s ',str);
                for tempk = 1:strLen
                    if strcmp(strCell{tempk},str)
                        break;
                    end
                end
                testseq(end+1) = tempk;
            end
            testsequence{targetIdx} = testseq;
            fprintf('\n');
            fprintf(frepId,'\n');
        end
        
        break;
    end
    
    
end