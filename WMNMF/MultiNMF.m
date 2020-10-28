function [U, V, centroidV, acc, nmi, Pi, Ri, Fi, ARi] = MultiNMF(X, K, gnd, options)
% This is a module of Multi-View Non-negative Matrix Factorization(MultiNMF)
%
% Notation:
% X ... a cell containing all views for the data
% K ... number of hidden factors
% gnd ... ground truth labels

viewNum = length(X);
Rounds = options.rounds;

U_ = [];
V_ = [];

U = cell(1, viewNum); % array
V = cell(1, viewNum);

j=0;
jj=0;
log = 0;
ac = 0;
nmi=0;
cnt=0;

% ==================initialization
while j < 3
    j = j + 1;
    if j == 1
        [U{1}, V{1}] = NMF(X{1}, K, options, U_, V_);
        %printResult(V{1}, gnd, K, options.kmeans);
    else
        [U{1}, V{1}] = NMF(X{1}, K, options, U_, V{viewNum});
        %printResult(V{1}, gnd, K, options.kmeans);        
    end
    for i = 2:viewNum
        [U{i}, V{i}] = NMF(X{i}, K, options, U_, V{i-1});
        %printResult(V{i}, gnd, K, options.kmeans);
    end
end

p=options.p;
centroidV = options.alpha(1)^p * V{1};
for i = 2:viewNum
    centroidV = centroidV + options.alpha(i)^p * V{i};
end
centroidV = centroidV / sum(options.alpha.^p);
% ====================================================
optionsForPerViewNMF = options;
oldL = 100; % the starting tolerance

tic  % used to start timer

l=zeros(1,Rounds);
while jj < Rounds %&& oldL>options.error %&& alphaerr>options.error
    jj = jj + 1;
    % ===================== update alpha ========================
    A=zeros(viewNum,1);
    for i=1:viewNum
        dV= V{i}-centroidV;
        A(i)=sum(sum(dV.^2));
    end
   
    dleft =sum((1./(A)).^(1/(p-1)));
    
    for i=1:viewNum
        dright=(A(i))^(1/(p-1));
        alpha(i)=1/dleft/dright;
    end
    alpha;
    % ================= update V* =======================
    centroidV = alpha(1)^p * V{1};
    for i = 2:viewNum
        centroidV = centroidV + alpha(i)^p * V{i};
    end
    centroidV = centroidV / sum(alpha.^p);
    %===========
    logL = 0;
    for i = 1:viewNum
        tmp1 = X{i} - U{i}*V{i}';
        tmp2 = V{i} - centroidV;
        logL = logL + sum(sum(tmp1.^2)) + alpha(i)^p * sum(sum(tmp2.^2));
    end
    
    l(jj)=logL;
    
    %logL % shows the obj for each iteration
    if(oldL < logL)
        U = oldU;
        V = oldV;
        alpha=oldalpha;
        centroidV=oldcentroidV;
        logL = oldL;
        jj = jj - 1;
        disp('restrart this iteration');
    else
        [acc, nmi, Pi, Ri, Fi, ARi] = Result(centroidV, gnd, K, options.kmeans);
    end
    
%     if (logL<options.error)
%         break
%     end
    
    oldU = U;
    oldV = V;
    oldalpha=alpha;
    oldcentroidV=centroidV;
    oldL = logL;
     % ============ PerViewNMF is used for update of u,v =======================
    for i = 1:viewNum
        optionsForPerViewNMF.alpha = alpha(i);
        [U{i}, V{i}] = PerViewNMF(X{i}, K, centroidV, optionsForPerViewNMF, U{i}, V{i});
    end    
end

toc
end