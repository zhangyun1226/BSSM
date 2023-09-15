temp = to'-to;
temp(temp<=0) = inf;
temp(isnan(temp)) = inf;
tC = 20;
tR = 300;
theta = 1;
tau = 4;
temp_delta = (1./(1+(temp-tC)./(theta*tR))).^tau;
delta = delta + temp_delta;     % 每新增一个句子，词对计数在此前的基础上累计

temp_eta = zeros(size(to));
temp_eta(to~=inf) = 1;
eta = eta + temp_eta;       % 每个单词出现的次数累计

w = theta * (eta .* delta) ./ (eta .* delta + eta.^2 - delta.^2 );  % 权重不累计，重新计算
w(isnan(w)) = 0;