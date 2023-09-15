input = [inf(5,1); 1*dT; inf; inf; 1; inf(6,1)];
tau_m = 100;
factor = exp( - 1 / tau_m );
Timeline = 1:T;
temp = Timeline - input;
temp(temp<0) = inf;
PSP = exp(-temp/tau_m);
V = zeros(N, T);
w = w - diag(diag(w)-1);
E_k = zeros(N, 1);
to = [];
for t = 1:1:T
    V(:, t) = w' * PSP(:, t) - E_k;
    row = find( V(:, t) >= theta );  
    if ~isempty(row)
        if length(theta) == N
            E_k(row) = E_k(row) + theta(row);      
        else
            E_k(row) = E_k(row) + theta;
        end    
        temp = Inf(N, 1);
        temp(row) = t;
        to = [to temp];
    end
end