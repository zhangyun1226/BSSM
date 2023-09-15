%% 建立working minicolumn set;
% column    -- 当前已有的微柱集合
% nc        -- 当前已有的微柱数量
% sentence  -- 当前处理句子
% tau_m     -- 膜电压时间常数
%%
WMS = zeros(length(sentence), 1);       % 记录当前句子对应的微柱index
for j = 1: length(sentence)
    new_word = string(sentence{j});
    p = [];
    %% 遍历所有已有微柱，看当前输入新词是否已经存在对应微柱
    for k = 1: nc
        if strcmp(column(k).word, new_word)
            p = k;
            break;
        end
    end
    %% 如果不存在，则建立新柱，进行初始化
    if isempty(p)
        nc = nc + 1;
%         column(nc) = initialcolumn(new_word, M, (j-1) * dT + 1);
        column(nc).word = new_word;
        for m = 1: M
            column(nc).neuron(m).precessor = [];
            column(nc).neuron(m).successor = [];
            column(nc).neuron(m).state = 1;  % 1代表UA，2代表PA，0代表不激活
            column(nc).neuron(m).time = j;      %(j-1) * dT + 1;
            column(nc).neuron(m).flag = 1;   % 1代表可搜索，0代表已被搜索过
            column(nc).neuron(m).eta = 0;   % 点火次数
            column(nc).neuron(m).path = [];   % 路径
        end
        WMS(j) = nc;
    else
        WMS(j) = p;
        if j == 1
            for m = 1: M
                column(p).neuron(m).state = 1; 
                column(p).neuron(m).time = j;%(j-1) * dT + 1;
            end
        else
            num_of_PA = 0;
            for m = 1: M
                presyn = column(p).neuron(m).precessor;
                if ~isempty(presyn)     % 有突触前神经元
                    ti = [];    
                    w = [];
                    for pre = 1: size(presyn, 1)
                        if column(presyn(pre,1)).neuron(presyn(pre,2)).state ~= 0  % 突触前神经元点火了
                            ti = [ti; column(presyn(pre,1)).neuron(presyn(pre,2)).time]; 
                            w = [w; presyn(pre, 4)];
                        end
                    end
                    if isempty(ti)
                        continue
                    end
                    [ti, order] = sort(ti);
                    w = w(order);
                    idx = find(ti==j);
                    ti(idx)=[];
                    w(idx) = [];
                    if isempty(w)
                        continue;
                    end
                    ti = dT(j-1) - dT(j-ti);
                    Timeline = 0: dT(j-1)+t_max;%(j-1) * dT + 1 + t_max; %  ti';
                    temp = Timeline - ti;
                    temp(temp <= 0) = Inf;
                    psp = Vnorm * ( exp( -temp / tau_m ) - exp( -temp / tau_s ) ); % = exp(- temp ./ tau_m);
                    membrane = w' * psp;
                    if any(membrane >= threshold)
                        column(p).neuron(m).state = 2;  % 成为PA
                        column(p).neuron(m).time = j;%(j-1) * dT + 1;
                        num_of_PA = num_of_PA + 1;
                    end
                end
            end
            
            if num_of_PA == 0
                for m = 1: M
                    column(p).neuron(m).state = 1;  
                    column(p).neuron(m).time = j;%(j-1) * dT + 1;
                end
            end
        end
    end
end
WM = column(WMS);