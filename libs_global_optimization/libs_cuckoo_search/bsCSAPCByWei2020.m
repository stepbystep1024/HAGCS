function [x,fval,exitFlag,output] = bsCSAPCByWei2020(objFunc, Lb, Ub, varargin)

    % parse some basic parameters for this process
    p = inputParser;
%     rng(125789);

    p = bsAddBasicalParameters(p, length(Lb));
    
    addParameter(p, 'nNest', 25 );     % The number of nests
    addParameter(p, 'pa', 0.5 );       % Discovery rate of alien eggs/solutions
    addParameter(p, 'beta', 3/2 );
    addParameter(p, 'alpha', 3/2 );
    % set default function of generation initial population as @bsGenerateInitialPopulationByLHS
    addParameter(p, 'initialPopulationFcn', @bsGenerateInitialPopulationByRandom );
    % whether to save the detail update information of all population. The
    % value is set to yes when we need to display an animation of
    % optimization process in general.
    addParameter(p, 'isSaveDetailUpdates', false);
%     addParameter(p, 'lambda', 6);
    addParameter(p, 'LM', 1000);
    
    p.parse(varargin{:});  
    params = p.Results;
    
    nNest = params.nNest;
    
    % call initial function to generate the initial population
    nests = params.initialPopulationFcn(Lb, Ub, nNest);
    
    
    
    % Get the current bestNest
    fitness = inf * ones(nNest, 1);
    [globalMinFVal, globalBestNest, nests, fitness] = bsGetBestNest(objFunc, nests, nests, fitness);
    xInit = globalBestNest;
    
    nfev = nNest;   %count the number of function evaluations
    
    fs = [0; 0; globalMinFVal];
    % track the convergence path of each nest
    detailUpdates = cell(1, nNest);
    
    if params.isSaveDetailUpdates
        for inest = 1 : nNest
            detailUpdates{inest} = [detailUpdates{inest}, [0; fitness(inest); nests(:, inest)]];
        end
    end
        
    %% Starting iterations
    nDim = size(nests, 1);
    Fm = 0.5;
    SF = [0.5];
    
    alpha_bank = zeros(1, params.LM);
    alpha_lastIndex = 0;
    is_alpha_full = 0;
    F_bank = zeros(1, params.LM);
    F_lastIndex = 0;
    is_F_full = 0;
    
    for iter = 1 : params.maxIter
        
        % 执行策略
        if is_alpha_full
            alpha = randn(1, nNest) * 0.2 + mean(alpha_bank);
            alpha(alpha<0) = 0.01;
        else
            alpha = 1.5 * rand(1, nNest);
        end
        
        newNests = bsGetCuckoos(params.beta, nests, globalBestNest, Lb, Ub, repmat(alpha, nDim, 1)); 
        [bestFVal, bestNest, nests, fitness, ~, isUpdate] = bsGetBestNestWithYesOrNo(objFunc, nests, newNests, fitness);
        [alpha_bank, alpha_lastIndex, is_alpha_full] = updateBank(alpha_bank, alpha_lastIndex, alpha(isUpdate==1), is_alpha_full);
        
        [newNests, F] = bsStrategy_2(nests, Fm, params.pa, Lb, Ub);
%         newNests = bsEmptyNests(nests, Lb, Ub, params.pa) ;
        [bestFVal, bestNest, nests, fitness, ~, isUpdate] = bsGetBestNestWithYesOrNo(objFunc, nests, newNests, fitness);
        
        % Update the counter
        nfev = nfev + 2 * nNest; 
        
        
        % 更新Fm
        if sum(isUpdate) > 0
%             SF = F(isUpdate == 1);
            [F_bank, F_lastIndex, is_F_full] = updateBank(F_bank, F_lastIndex, F(isUpdate==1), is_F_full);
        end
        
        if is_F_full
            wf = 0.8 + 0.2 * rand();
            mean_SF = sum(F_bank.^2) / sum(F_bank);
            Fm = wf * Fm + (1 - wf) * mean_SF;   
        end
        
        % 更新Best
        if bestFVal < globalMinFVal
            globalMinFVal = bestFVal;
            globalBestNest = bestNest;
        end
        
        if params.isSaveMiddleRes
            fs = [fs, [iter; nfev; globalMinFVal]];
        end
        
        if params.isSaveDetailUpdates
            for inest = 1 : nNest
                detailUpdates{inest} = [detailUpdates{inest}, [0; fitness(inest); nests(:, inest)]];
            end
        end
        
        data.fNew = globalMinFVal;
        data.nfev = nfev;
        data.iter = iter;
        
        exitFlag = bsCheckStopCriteria(data, params);
        [data] = bsPlotMidResults(xInit, data, params, Lb, Ub, exitFlag > 0);
        
        if exitFlag > 0
            break;
        end
        
        
    end %% End of iterations

    x = globalBestNest;
    fval = globalMinFVal;
    output.funcCount = nfev;
    output.iterations = iter;
    output.midResults = fs;
    output.frames = data.frames;
    output.detailUpdates = detailUpdates;
end

function [bestFVal, bestNest, nests, fitness, K, isUpdate] = bsGetBestNestWithYesOrNo(fobj, nests, newNest, fitness)
    isUpdate = zeros(1, size(nests, 2));
    
    for j = 1 : size(nests, 2)
        
        fNew = fobj(newNest(:, j));
        
        if fNew <= fitness(j)
           fitness(j) = fNew;
           nests(:, j) = newNest(:, j);
           isUpdate(j) = 1;
        end
    end
    
    % Find the current bestNest
    [bestFVal, K] = min(fitness) ;
    bestNest = nests(:, K);
end

function [bank, lastIndex, is_full] = updateBank(bank, lastIndex, newData, is_full)
    n1 = length(bank) - lastIndex;
    nData = length(newData);
    
    if n1 > nData
        n1 = nData;
        bank(lastIndex+1:lastIndex+n1) = newData;
        lastIndex = lastIndex+n1+1;
    else
        n2 = nData - n1;
        bank(lastIndex+1:end) = newData(1:n1);
        bank(1:n2) = newData(n1+1:end);
        lastIndex = n2;
        is_full = 1;
    end
    
end

% function F = bsGenF(Fm, nNest)
%     
%     F = random(pd,1, nNest);
% end

% function newNests = bsStrategy_1(nests, bestNest, beta, alpha)
% 
%     % Levy flights
%     newNests = nests;
%     [nDim, nNest] = size(newNests);
%     steps = bsLevy(nDim, nNest, beta);
%     newNests = newNests + alpha .* steps .* (newNests - repmat(bestNest, 1, nNest));
% %     for j = 1 : nNest
% %         nest = newNests(:, j);
% %         step = steps(:, j);
% %  
% %         stepsize = alpha .* step .* (nest - bestNest);
% %         newNests(:, j) = nest + stepsize;
% %     end
% end

function [ out ] = bsRandK(data, k)
    n = length(data);
    repeat_times = ceil(k/n);
    index = [];
    for i = 1 : repeat_times
        index = [index, randperm(n)];
    end

    out = data(index(1:k));
end

function [newNests, F] = bsStrategy_2(nests, Fm, pa, Lb, Ub)
    [nDim, nNest] = size(nests);
    pd = makedist('tLocationScale','mu', Fm, 'sigma',0.1, 'nu', 1);
    
%     F = zeros(1, nNest);
    F = random(pd, 1, nNest);
    F(F>1) = 1;
    F(F<0) = 0.05;
%     for i = 1 : nNest
%         F(i) = random(pd);
%         
%         while F(i) < 0
%             F(i) = random(pd);
%         end
%         
%         if F(i) > 1
%             F(i) = 1;
%         end
%     end
    
%     F = bsGenF(Fm, nNest);
    
    K = rand(size(nests)) <= pa;
    stepsize = repmat(F, nDim, 1) .* (nests(:, randperm(nNest)) - nests(:, randperm(nNest)));
    newNests = nests + stepsize .* K;
    
    newNests = bsSimpleBounds(newNests, Lb, Ub);
end

