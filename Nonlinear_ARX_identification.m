% Clear all variables in the workspace
clear all

% Model parameters
na = 1;     % Number of AR (AutoRegressive) coefficients
nb = na;    % Number of MA (Moving Average) coefficients
m = 3;      % Maximum degree of polynomials



% Generate the t matrix representing all possible combinations of coefficients
t = zeros((m + 1)^(na + nb), na + nb);

for c = 1:(na + nb)
    line = 1;
    step = (m + 1)^(c - 1);
    for a = 1:(m + 1)^(na + nb - 1) / step
        for k = 0:m
            for b = line:(line + step)
                t(b, c) = k;
            end
            line = line + step;
        end
    end
end

t(end, :) = [];

j = (m + 1)^(na + nb);

% Remove rows from t where the sum of coefficients exceeds the maximum degree
while (j >= 1)
    if (sum(t(j, :)) > m)
        t(j, :) = [];
        j = j - 1;
    else
        j = j - 1;
    end
end

%% Load data from the 'iddata-01.mat' file
load("iddata-01.mat");

% Extract data from the 'id' dataset
y_id = id.OutputData;
u_id = id.InputData;

% Extract data from the 'val' dataset
y_val = val.OutputData;
u_val = val.InputData;

%% Build the phi matrix for identification
N = length(y_id);
phi = ones(N, length(t));
a = length(t);

for k = 1:N
    for j = 1:na
        for i = 1:a
            if (k - j) <= 0
                phi(k, i) = 0;
            else
                phi(k, i) = phi(k, i) * ((y_id(k - j))^t(i, j)) * (u_id(k - j)^t(i, j + na));
            end
        end
    end
end

phi(1, :) = 1;
theta = phi \ y_id;

y_hat_id = phi * theta;

%% Build the phi matrix for validation
Nval = length(y_val);
phiv = ones(Nval, a);

for k = 1:Nval
    for j = 1:na
        for i = 1:a
            if (k - j) <= 0
                phiv(k, i) = phiv(k, i);
            else
                phiv(k, i) = phiv(k, i) * ((y_val(k - j))^t(i, j)) * (u_val(k - j)^t(i, j + na));
            end
        end
    end
end
phiv(1,:) = 1;
y_hat_val = phiv * theta;

%% Simulate the identified model for identification data
ysimid = zeros(N, 1);
for k = 1:N
    d = ones(1, length(t));
    for j = 1:na
        for i = 1:a
            if (k - j) <= 0
                d(i) = ysimid(1);
            else
                d(i) = d(i) * ((ysimid(k - j))^t(i, j)) * (u_id(k - j)^t(i, j + na));
            end
        end
    end
    ysimid(k) = d * theta;
    clear d;
end

%% Simulate the identified model for validation data
ysimval = zeros(Nval, 1);
for k = 1:Nval
    d = ones(1, length(t));
    for j = 1:na
        for i = 1:a
            if (k - j) <= 0
                d(i) = ysimval(1);
            else
                d(i) = d(i) * ((ysimval(k - j))^t(i, j)) * (u_val(k - j)^t(i, j + na));
            end
        end
    end
    ysimval(k) = d * theta;
    clear d;
end

%% plots
subplot(221);
plot(y_hat_id);
hold
plot(y_id);
legend('Yhat Identification','Y identification');
title("Y Identification VS Yhat Identification");
xlabel('Time(seconds)');
ylabel('Amplitude');

subplot(222);
plot(y_hat_val);
hold
plot(y_val);
legend('Yhat validation','Y Validation');
title("Y Validation VS Yhat Validation");
xlabel('Time(seconds)');
ylabel('Amplitude');


subplot(223);
simid = iddata(ysimid,u_id,id.Ts);
compare(simid,id);
legend('Y Simulated Identification','Y Identification');
title("Y Identification VS Y Simulated Identification");
xlabel('Time');
ylabel('Amplitude');

subplot(224);
simval = iddata(ysimval,u_val,val.Ts);
compare(simval,val);
legend('Y simulated for Validation','Y Validation');
title("Y Validation vs Y Simulated Validation");
xlabel('Time');
ylabel('Amplitude');

%% Define the maximum rank and calculate the MSE for polynomial degree optimization
m = 5;
na = 3;

% Initialize arrays to store MSE values for different polynomial ranks
mse_id = zeros(m, na);
mse_id_hat = zeros(m, na);
mse_val = zeros(m, na);
mse_val_hat = zeros(m, na);

% Loop through different polynomial ranks
for l = 1:m
    for n = 1:na

        % Generate matrix t for the current rank
        t = zeros((l + 1)^(n + n), n + n);

        for c = 1:(n + n)
            line = 1;
            step = (l + 1)^(c - 1);
            for a = 1:(l + 1)^(n + n - 1) / step
                for k = 0:l
                    for b = line:(line + step)
                        t(b, c) = k;
                    end
                    line = line + step;
                end
            end
        end

        t(end, :) = [];

        j = (l + 1)^(n + n);

        % Remove rows from t where the sum of coefficients exceeds the maximum degree
        while (j >= 1)
            if (sum(t(j, :)) > l)
                t(j, :) = [];
                j = j - 1;
            else
                j = j - 1;
            end
        end

        % Build phi matrix for identification with the new t matrix
        N = length(y_id);
        phi = ones(N, length(t));
        a = length(t);

        for k = 1:N
            for j = 1:n
                for i = 1:a
                    if (k - j) <= 0
                        phi(k, i) = 0;
                    else
                        phi(k, i) = phi(k, i) * ((y_id(k - j))^t(i, j)) * (u_id(k - j)^t(i, j + n));
                    end
                end
            end
        end

        phi(1, :) = 1;
        theta = phi \ y_id;
        y_hat_id = phi * theta;

        % Build phi matrix for validation with the new t matrix
        Nval = length(y_val);
        phiv = ones(Nval, a);

        for k = 1:Nval
            for j = 1:n
                for i = 1:a
                    if (k - j) <= 0
                        phiv(k, i) = phiv(k, i);
                    else
                        phiv(k, i) = phiv(k, i) * ((y_val(k - j))^t(i, j)) * (u_val(k - j)^t(i, j + n));
                    end
                end
            end
        end

        y_hat_val = phiv * theta;

       ysimid = zeros(N,1);
d = ones(1,length(t));

for k = 1:N
    d = ones(1,length(t));
    for j = 1:n
        for i = 1:a
            if (k-j) <= 0
                d(i) = ysimid(1);
            else
                d(i) = d(i)*((ysimid(k-j))^t(i,j))*(u_id(k-j)^t(i,j+n));
            end
        end
    end
    ysimid(k) = d*theta;
    clear d;
end

ysimval = zeros(N,1);
for k = 1:Nval
    d = ones(1,length(t));
    for j = 1:n
        for i = 1:a
            if (k-j) <= 0
                d(i) = ysimval(1);
            else
                d(i) = d(i)*((ysimval(k-j))^t(i,j))*(u_val(k-j)^t(i,j+n));
            end
        end
    end
    ysimval(k) = d*theta;
    clear d;
end

    % Calculate the MSE and store it into a vector
    mse_id(l,n) = mean((y_id(:) - ysimid(:)).^2);
    mse_id_hat(l,n) = mean((y_id(:) - y_hat_id(:)).^2);
    mse_val(l,n) = mean((y_val(:) - ysimval(:)).^2);
    mse_val_hat(l,n) = mean((y_val(:) - y_hat_val(:)).^2);
    end
end


%% plot MSE
figure;
% Plot the points for Identification MSE
subplot(221);
scatter3(repmat(1:m, 1, na), kron(1:na, ones(1, m)), mse_id(:));
grid on;
title("MSE for Identification");
xlabel('Polynomial Degree');
ylabel('Number of AutoRegressive coefficients');
zlabel('MSE');

% Plot the points for Validation MSE
subplot(222);
scatter3(repmat(1:m, 1, na), kron(1:na, ones(1, m)), mse_val(:));
grid on;
title("MSE for Validation");
xlabel('Polynomial Degree');
ylabel('Number of AutoRegressive coefficients');
zlabel('MSE');

% plot MSE hat
% Plot the points for Identification MSE Hat
subplot(223);
scatter3(repmat(1:m, 1, na), kron(1:na, ones(1, m)), mse_id_hat(:));
grid on;
title("MSE Hat for Identification");
xlabel('Polynomial Degree');
ylabel('Number of AutoRegressive coefficients');
zlabel('MSE Hat');

% Plot the points for Validation MSE Hat
subplot(224);
scatter3(repmat(1:m, 1, na), kron(1:na, ones(1, m)), mse_val_hat(:));
grid on;
title("MSE Hat for Validation");
xlabel('Polynomial Degree');
ylabel('Number of AutoRegressive coefficients');
zlabel('MSE Hat');
