function [L, P] = lqr_infinite_horizon_solution(Q, R)

    %% find the infinite horizon L and P through running LQR back-ups
    %%   until norm(L_new - L_current, 2) <= 1e-4  
    dt = 0.1;
    mc = 10; mp = 2.; l = 1.; g= 9.81;

    % TODO write A,B matrices
    a1 = mp * g / mc; 
    a2 = (mc + mp) * g / l / mc;

    A = eye(4) + dt * [0 0 1 0; 0 0 0 1; 0 a1 0 0; 0 a2 0 0];
    B = dt * [0; 0; 1 / mc; 1/ (l * mc)];

    % TODO implement Riccati recursion
    xDim = size(A,1)
    P0 = zeros(xDim,xDim)
    P_current = P0;   

    L_current = -inv(R + B.' * P_current * B) * B.' * P_current * A;
    L_new = L_current;
    
    P_new = Q + L_new.' * R * L_new + (A + B * L_new).' * P_current * (A + B * L_new);


    while (1)
        L_current = L_new;
        P_current = P_new;
        L_new = -inv(R + B.' * P_current * B) * B.' * P_current * A;
        P_new = Q + L_new.' * R * L_new + (A + B * L_new).' * P_current * (A + B * L_new);
        if (norm(L_new - L_current, 2) <= 1e-4)
            break;
        end
    end
    L = L_new
    P = P_new
end