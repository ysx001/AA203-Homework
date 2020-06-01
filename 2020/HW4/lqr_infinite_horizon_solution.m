function [L, P] = lqr_infinite_horizon_solution(Q, R)

    %% find the infinite horizon L and P through running LQR back-ups
    %%   until norm(L_new - L_current, 2) <= 1e-4  
    dt = 0.1;
    mc = 10; mp = 2.; l = 1.; g= 9.81;

    % TODO write A,B matrices
    a1 = dt*(mp*g)/mc; 
    a2 = dt*((mc + mp)*g)/(l*mc);

    A = [1 0 dt 0; 0 1 0 dt; 0 a1 1 0; 0 a2 0 1];
    B = [0; 0; dt/mc; dt/(mc*l)];

    % TODO implement Riccati recursion
    n = size(A,1);
    P = zeros(n,n);
    BT = transpose(B);
    AT = transpose(A);
    L = -inv(R+ BT * P * B)*(BT*P*A);  

    L_old = ones(2,4);

    while norm(L - L_old,2) > 1e-4
        P = AT*P*A - (AT*P*B)*inv(R + BT*P*B)*(BT*P*A) + Q;
        L_old = L;
        L = -inv(R+ BT * P * B)*(BT*P*A);
    end
end