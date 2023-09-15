%% ����working minicolumn set;
% column    -- ��ǰ���е�΢������
% nc        -- ��ǰ���е�΢������
% sentence  -- ��ǰ�������
% tau_m     -- Ĥ��ѹʱ�䳣��
%%
WMS = zeros(length(sentence), 1);       % ��¼��ǰ���Ӷ�Ӧ��΢��index
for j = 1: length(sentence)
    new_word = string(sentence{j});
    p = [];
    %% ������������΢��������ǰ�����´��Ƿ��Ѿ����ڶ�Ӧ΢��
    for k = 1: nc
        if strcmp(column(k).word, new_word)
            p = k;
            break;
        end
    end
    %% ��������ڣ��������������г�ʼ��
    if isempty(p)
        nc = nc + 1;
%         column(nc) = initialcolumn(new_word, M, (j-1) * dT + 1);
        column(nc).word = new_word;
        for m = 1: M
            column(nc).neuron(m).precessor = [];
            column(nc).neuron(m).successor = [];
            column(nc).neuron(m).state = 1;  % 1����UA��2����PA��0��������
            column(nc).neuron(m).time = j;      %(j-1) * dT + 1;
            column(nc).neuron(m).flag = 1;   % 1�����������0�����ѱ�������
            column(nc).neuron(m).eta = 0;   % ������
            column(nc).neuron(m).path = [];   % ·��
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
                if ~isempty(presyn)     % ��ͻ��ǰ��Ԫ
                    ti = [];    
                    w = [];
                    for pre = 1: size(presyn, 1)
                        if column(presyn(pre,1)).neuron(presyn(pre,2)).state ~= 0  % ͻ��ǰ��Ԫ�����
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
                        column(p).neuron(m).state = 2;  % ��ΪPA
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