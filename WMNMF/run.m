clear all;
close all;
clc
rng(20);
%%
%
% for synthetic data: no error bound; kmeans=1
% for digit data: set error 1.39051e-04; kmeans=1
% for reuters data: error 1.39051e-04; kmeans=1
% for synthetic data: error 1.39051e-04; kmeans=1
% parameter setting
options = [];
options.maxIter = 200;
options.error = 1.39051e-04;
options.nRepeat = 30; % used for iteration of U V - PerViewNMF
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 2; % used for iteration of objective - multiNMF
options.p=5;
options.kmeans = 1; % lite kmeans for comparison
rep=1;

load ../handwritten.mat
load ../uci-digit.mat
data{1} = zer';
data{2}=mfeat_fou';
data{3}=mfeat_fac';
data{4} = pixel';
K = 10;
gnd=gnd+1;

% normalize data matrix
for i = 1:length(data)
    data{i} = data{i} / sum(sum(data{i}));
end

% options alpha is an array of weights for different views
num=length(data);
options.alpha = ones(1,num)/num;

for i = 1:rep
   [U, V, centroidV, acc(i), nmi(i), Pi(i), Ri(i), Fi(i), ARi(i)] = MultiNMF(data, K, gnd, options);
end

%% result
[mean(acc) std(acc)]
[mean(nmi) std(nmi)]
[mean(Pi) std(Pi)]
[mean(Ri) std(Ri)]
[mean(Fi) std(Fi)]
[mean(ARi) std(ARi)]







