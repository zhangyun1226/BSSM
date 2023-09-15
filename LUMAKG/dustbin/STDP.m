temp = to'-to;
temp(temp<=0) = inf;
temp(isnan(temp)) = inf;
tau = 150;
deltaw = exp(-(temp-dT)/tau);
delta = delta + deltaw;

temp_eta = zeros(size(to));
temp_eta(to~=inf) = 1;
eta = eta + temp_eta;

w = theta * (eta .* delta) ./ (eta .* delta + eta.^2 - delta.^2 ); 
w(isnan(w)) = 0;
