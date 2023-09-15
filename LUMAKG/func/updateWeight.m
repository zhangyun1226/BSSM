%% Updating weights
% final_path -- 找到的最终链路
% column
theta = 1;
tC = 20;
tR = 300;
tau = 4;
%% 有的相邻PA之间此前未建立过连接
for i = 1: length(final_path) - 1
    p = final_path(i, 1);
    m = final_path(i, 2);
    for j = i+1: length(final_path)
        q = final_path(j, 1);
        n = final_path(j, 2);
        if isempty(column(p).neuron(m).successor) || ~issubset([q n], column(p).neuron(m).successor(:, 1:2))
            column(p).neuron(m).successor = [column(p).neuron(m).successor; q n 0 0];
            column(q).neuron(n).precessor = [column(q).neuron(n).precessor; p m 0 0];
        end
    end
end
sentence_id = [final_path, (1:10)'];
%% 调整权重
for i = 1: length(final_path)
    p = final_path(i, 1);
    m = final_path(i, 2);
    column(p).neuron(m).eta = column(p).neuron(m).eta + 1;      % 神经元点火次数
    eta = column(p).neuron(m).eta;
%     for j = i+1: 10
%         [~, row] = issubset(sentence_id(j, 1:2), column(p).neuron(m).successor(:, 1:2));
%         if ~isempty(row)
%             q = column(p).neuron(m).successor(row, 1);
%             n = column(p).neuron(m).successor(row, 2);
%             [~, row2] = issubset([p m], column(q).neuron(n).precessor(:, 1:2));
%             delta = column(q).neuron(n).precessor(row2, 3);
    for j = 1: size(column(p).neuron(m).successor, 1)
        q = column(p).neuron(m).successor(j, 1);
        n = column(p).neuron(m).successor(j, 2);
        [~, row] = issubset([p m], column(q).neuron(n).precessor(:, 1:2));
        delta = column(q).neuron(n).precessor(row, 3);
        [~, loc] = issubset([q n], sentence_id(:,1:2));    % (p,m)的后继属于当前WMS, 更新delta, 否则直接更新w
        if ~isempty(loc)
            loc(loc<=i)=[];
            loc(loc>i+8)=[];
            if isempty(loc)
                continue;
            end
            timediff = loc - i;
            temp = dT(timediff);
            delta = delta + sum((1./(1+(temp-tC)./(theta*tR))).^tau);
            column(q).neuron(n).precessor(row, 3) = delta;
            column(p).neuron(m).successor(j, 3) = delta;  
        end
        w = theta * (eta .* delta) ./ (eta .* delta + eta.^2 - delta.^2 );
        column(q).neuron(n).precessor(row, 4) = w;
        column(p).neuron(m).successor(j, 4) = w;  
    end
end