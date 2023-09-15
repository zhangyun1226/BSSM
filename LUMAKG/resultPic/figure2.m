clear
eta = 1;
tau = 100;
decay_s = .98;
decay_m = .995;
T = 200;
w_max = 1;
threshold = 1;
num = 5;
tmax = 1000;
input_num = 4;
retr_num = 2;
ti = ceil((0:1:num-1)' * 200 + 5 + 30 * rand(num,1));
% w = .1*rand(num,num) * w_max .* tril(ones(num,num),-1);
w =.1 * ones(num,num) * w_max .* tril(ones(num,num),-1);
timeline = 1:1:tmax;

for epoch = 1:30
    dt = ti - ti';
    dt(dt<=0)=inf;
    dw = exp(-dt/tau) .* w .* (w_max - w);
    w = w + eta * dw;
end
ti(input_num+1:num)=inf;
v_input = zeros(num, tmax);
r = 1:5;
for n = 1:input_num
    v_input(r(n),ti(n))=1.5;
end

v_input(4,:)=0;

v_background = -.51 * sin(2 * pi * timeline/T);
v_inhibit = zeros(num, tmax);
v_context = zeros(num, tmax);
v = zeros(num, tmax);
to = inf * ones(num, tmax);
depolar = zeros(num,tmax);

spike = zeros(num,1);
v_tau_m = zeros(num,1);
v_tau_s = zeros(num,1);

for timestep = 1:tmax
    idx = find(ti==timestep);
    if idx > 0
        s = (timestep:tmax) - timestep;
        s = repmat(s,num,1);
        tmp = -2 * exp(-s/50);
        if idx < input_num
            tmp(idx,:) = 0;
        end
        v_inhibit(:,timestep:tmax) = v_inhibit(:,timestep:tmax) + tmp;
    end
    
    v_tau_m = (1 - spike) .* v_tau_m * decay_m + w * spike;
    v_tau_s = (1 - spike) .* v_tau_s * decay_s + w * spike;
    tmp = (v_tau_m - v_tau_s) * 1.5;
    v_context(:,timestep) = tmp;
    
    for n = 1: num
        if tmp(n) > .5
            depolar(n, timestep: min(timestep+1.5*T, tmax)) = .5;
        end
    end

    v(:,timestep) = v_input(:,timestep) + v_background(:,timestep) + depolar(:,timestep) + v_inhibit(:,timestep);
    [vm,idx] = max(v(:,timestep));
    if vm >= threshold
        to(idx,timestep) = timestep;
        spike(idx)=1;
        depolar(idx,timestep:tmax) = 0;
    else
        spike = zeros(num,1);
    end
end
save figure2.mat v_background v_context depolar v_inhibit v ti
