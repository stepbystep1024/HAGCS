function [x,fval,exitFlag,output] = bsMSACSByGao2021(objFunc, Lb, Ub, varargin)

    % parse some basic parameters for this process
    p = inputParser;
%     rng(125789);

    p = bsAddBasicalParameters(p, length(Lb));
    
    addParameter(p, 'nNest', 25 );     % The number of nests
    addParameter(p, 'pa', 0.5 );       % Discovery rate of alien eggs/solutions
    addParameter(p, 'beta', 3/2 );
    addParameter(p, 'alpha', 0.1 );
    % set default function of generation initial population as @bsGenerateInitialPopulationByLHS
    addParameter(p, 'initialPopulationFcn', @bsGenerateInitialPopulationByRandom );
    % whether to save the detail update information of all population. The
    % value is set to yes when we need to display an animation of
    % optimization process in general.
    addParameter(p, 'isSaveDetailUpdates', false);
    addParameter(p, 'lambda', 6);
    addParameter(p, 'LM', 2000);
    
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
    best_nest_bank = zeros(nDim, params.LM);
    m_bank = zeros(5, params.LM);
    n_bank = zeros(5, params.LM);
    selected_index = zeros(1, nNest);
    best_nest_bank(:, 1) = globalBestNest;
    alpha = params.alpha;
    
    
    for iter = 1 : params.maxIter
        
        bank_index = mod(iter-1, params.LM) + 1;
        m_bank(:, bank_index) = 0;
        n_bank(:, bank_index) = 0;
        
        if iter <= params.LM
            SP = ones(1, nNest) * 0.2;
        else
            SP = bsGenSP(n_bank, m_bank);
        end
        
        % 根据SP选择策略
        p = rand(1, nNest);
        for i = 1 : nNest
            for j = 1 : 5
                if p(i) < sum(SP(1:j))
                    selected_index(i) = j;
                    m_bank(j, bank_index) = m_bank(j, bank_index) + 1;
                    break;
                end
            end
        end
        
        newNests = nests;
        
        % 执行策略
        index = selected_index == 1;
        newNests(:, index) = bsStrategy_1(index, nests, globalBestNest, params.beta, alpha);
        
        index = selected_index == 2;
        newNests(:, index) = bsStrategy_2(index, nests);
        
        index = selected_index == 3;
        newNests(:, index) = bsStrategy_3(index, nests, globalBestNest);
        
        index = selected_index == 4;
        newNests(:, index) = bsStrategy_4(index, nests, globalBestNest, best_nest_bank(:, 1:min(params.LM, iter)));
        
        index = selected_index == 5;
%         newNests(:, index) = bsStrategy_5(index, nests, globalBestNest, params.lambda);
        newNests(:, index) = bsStrategy_2(index, nests);
        
        newNests = bsSimpleBounds(newNests, Lb, Ub);
        
        % Update the counter
        nfev = nfev + nNest; 
          
        % Discovery and randomization
%         newNests = bsEmptyNests(nests, Lb, Ub, params.pa) ;

        % Evaluate this set of solutions
        [bestFVal, bestNest, nests, fitness, ~, isUpdate] = bsGetBestNestWithYesOrNo(objFunc, nests, newNests, fitness);
        
        % 更新np
        for i = 1 : 5
            index = selected_index == i;
            n_bank(i, bank_index) = sum(isUpdate(index));
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

function SP = bsGenSP(n_bank, m_bank)
    SP = zeros(1, 5);
    n_bank = sum(n_bank, 1);
    m_bank = sum(m_bank, 1);
    
    for j = 1 : 5
        if n_bank == 0
            SP(j) = 0.01;
        else
            SP(j) = n_bank(j) / m_bank(j);
        end
    end
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

function F = bsGenF(nNest, nDim)
    F = repmat(0.5 + 0.3 * randn(1, nNest), nDim, 1);
    F(F<0) = 0.001;
end

function newNests = bsStrategy_1(index, nests, bestNest, beta, alpha)

    % Levy flights
    newNests = nests(:, index);
    [nDim, nNest] = size(newNests);
    steps = bsLevy(nDim, nNest, beta);
    newNests = newNests + alpha .* steps .* (newNests - repmat(bestNest, 1, nNest));
    
%     for j = 1 : nNest
%         nest = newNests(:, j);
%         step = steps(:, j);
%  
%         stepsize = alpha .* step .* (nest - bestNest);
%         newNests(:, j) = nest + stepsize;
%     end
end

function [ out ] = bsRandK(data, k)
    n = length(data);
    repeat_times = ceil(k/n);
    index = [];
    for i = 1 : repeat_times
        index = [index, randperm(n)];
    end

    out = data(index(1:k));
end

function newNests = bsStrategy_2(index, nests)
    newNests = nests(:, index);
    nPreNest = size(nests, 2);
    [nDim, nNest] = size(newNests);
    
    F = bsGenF(nNest, nDim);
    R1 = bsRandK(1:nPreNest, nNest);
    R2 = bsRandK(1:nPreNest, nNest);
    
    newNests = newNests + F .* (nests(:, R1) - nests(:, R2));
end

function newNests = bsStrategy_3(index, nests, bestNest)
    newNests = nests(:, index);
    nPreNest = size(nests, 2);
    [nDim, nNest] = size(newNests);
    
    F = bsGenF(nNest, nDim);
    R1 = bsRandK(1:nPreNest, nNest);
    R2 = bsRandK(1:nPreNest, nNest);
    R3 = bsRandK(1:nPreNest, nNest);
    R4 = bsRandK(1:nPreNest, nNest);
    
    newNests = bestNest + F .* (nests(:, R1) - nests(:, R2)) + F .* (nests(:, R3) - nests(:, R4));
end

function newNests = bsStrategy_4(index, nests, bestNest, bank)
    newNests = nests(:, index);
    nPreNest = size(nests, 2);
    [nDim, nNest] = size(newNests);
    
    F = bsGenF(nNest, nDim);
    R1 = bsRandK(1:nPreNest, nNest);
    RBank = bsRandK(1:size(bank, 2), nNest);
    
    newNests = bestNest + F .* (bank(:, RBank) - nests(:, R1)) + F .* (bank(:, RBank) - repmat(bestNest, 1, nNest));
end

function newNests = bsStrategy_5(index, nests, bestNest, lambda)
    phi = 2 / (2 - lambda - sqrt(lambda^2 - 4 * lambda));
    
    newNests = nests(:, index);
    [nDim, nNest] = size(newNests);
    
    F = repmat( rand(1, nNest), nDim, 1);
    newNests = phi * lambda * (F .* (newNests - bestNest));
end


