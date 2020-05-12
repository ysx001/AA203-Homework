n = 20;

delta = 10;

gamma = 0.95;

eye = [15, 15];

goal = [19, 9];

 

%

P = zeros(n, n);

for i = 1 : n

    for j = 1 : n

        d = (i-eye(1))^2 + (j-eye(2))^2;

        P(i, j) = exp(-d/2/delta/delta);

    end

end

 

 

% value iteration

V = zeros(n, n);

for i = 1 : 1000

    [V, A] = update_V(V, P, gamma, goal);

end

 

% part b

s = [9, 19];

x = [];

y = [];

while s(1) ~= goal(1) || s(2) ~= goal(2)

    x = [x; s(1)];

    y = [y; s(2)];

    if A(s(1), s(2)) == 1

        s = [max(s(1)-1, 1), s(2)];

    elseif A(s(1), s(2)) == 2

        s = [min(s(1)+1, size(A, 1)), s(2)];

    elseif A(s(1), s(2)) == 3

        s = [s(1), max(s(2)-1, 1)];

    elseif A(s(1), s(2)) == 4

        s = [s(1), min(s(2)+1, size(A, 2))];

    end

end

 

path = zeros(size(A));

for i = 1 : length(x)

    path(x(i), y(i)) = 1;

end

 

%

figure,subplot(121),

imshow(V, []),colormap jet, title('Heatmap of the value function')

subplot(122),imshow(path, []), title('Trajectory')

%

 

function [V, A] = update_V(V, P, gamma, goal)

for i = 1 : size(V, 1)

    for j = 1 : size(V, 2)

        [V(i, j), A(i, j)] = update_pixel(V, i, j, P(i, j), gamma, goal);

    end

end

end

 

function [v, a] = update_pixel(V, i, j, p, gamma, goal)

 

   

    xs = [max(i-1, 1), min(i+1, size(V, 1)), i, i]; % up down left right

    ys = [j, j, max(j-1, 1), min(j+1, size(V, 2))];

    bonus = [0, 0, 0, 0];

    for n = 1 : 4

        if xs(n) == goal(1) && ys(n) == goal(2)

            bonus(n) = 1;

        end

    end

   

    % update v without considering p now

    v = [0, 0, 0, 0];  

    for n = 1 : 4

        v(n) = V(xs(n), ys(n));

    end

    v = gamma * v + bonus;

    [mv, a] = max(v);

    v = mean(v) * p + (1 - p) * mv;

end