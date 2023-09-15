%% testing
time = [];
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
        time = [time dT(j)];
        for m = 1: M
            column(p).neuron(m).state = 1; 
            column(p).neuron(m).time = j;%(j-1) * dT + 1;
        end
    end
end
path = cell(M, 1);      % 初始路径
for i = 1: M
    path{i} = [WMS(1) i];
end
for epo = 2: 30
    np = 0;         % 新一轮路径数量
    new_path = [];
    for i = 1: length(path)
        WM = path{i};
        suc = column(WM(end, 1)).neuron(WM(end, 2)).successor;
        if isempty(suc)   % 该路径已终止，输出该句子
            for j = 1: size(WM, 1)
                fprintf('%s ', column(WM(j, 1)).word);
            end
            fprintf('\n');
            continue;
        else
            for j = 1: size(suc, 1)
                presyn = column(suc(j,1)).neuron(suc(j,2)).precessor;
                ti = [];    w = [];
                for pre = 1: size(presyn, 1)
                    if issubset(presyn(pre, 1:2), WM)  % 突触前神经元点火了
                        ti = [ti; column(presyn(pre,1)).neuron(presyn(pre,2)).time]; 
                        w = [w; presyn(pre, 4)];
                    end
                end
                if isempty(ti)
                    continue
                end
                [ti, order] = sort(ti);
                w = w(order);
                ti = dT(epo-1) - dT(epo-ti);
                Timeline = 0: dT(epo-1) + t_max;%(epo-1) * dT + 1 + t_max + 1; %  ti';
                temp = Timeline - ti;
                temp(temp <= 0) = Inf;
                psp = Vnorm * ( exp( -temp / tau_m ) - exp( -temp / tau_s ) ); % = exp(- temp ./ tau_m);
                membrane = w' * psp;
                col = find( membrane >= theta );  
                if ~isempty(col) %&& Timeline(col) < (epo-1) * dT + 1
                    column(suc(j,1)).neuron(suc(j,2)).state = 2;
                    column(suc(j,1)).neuron(suc(j,2)).time = epo;%(epo-1) * dT + 1;
                    np = np + 1;
                    new_path{np} = [WM; suc(j,1:2)];
                end
            end
        end
        if epo <= length(WMS)
            for m = 1: M
                if issubset([WMS(epo) m], suc(:, 1:2)) && column(WMS(epo)).neuron(m).state ~= 2
                    np = np + 1;
                    new_path{np} = [WM; WMS(epo) m];
                end
            end
        end
    end
    if np == 0
        break;
    end
    path = new_path; 
    for i = 1: length(path)
        for j = 1: size(path{i}, 1)
            fprintf('%s ',column(path{i}(j, 1)).word);
        end
        fprintf('\n');
    end
end